import os
import time
import torch
# import librosa
import argparse
import numpy as np
import torch.nn as nn
# import IPython.display as ipd
# from matplotlib import pyplot as plt
from test_model import test_model
from asteroid.models import ConvTasNet
from datasets import MUSDB_data, Slakh_data
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

import utils

os.environ['TZ'] = 'EST+05EDT,M4.1.0,M10.5.0'
time.tzset()


parser = argparse.ArgumentParser(description='train_model')

parser.add_argument('--lr', type=float, default=0.001,  help='Number of sources')
parser.add_argument('--max_epoch', type=int, default=100,  help='Input length to the network')
parser.add_argument('--batch', type=int, default=2,  help='Batch size to generate data')
parser.add_argument('--n_src', type=int, default=2,  help='Number of sources/masks')
parser.add_argument('--sample_rate', type=int, default=44100,  help='Sampling rate')
parser.add_argument('--weight_src', type=float, default=1,  help='Weight of loss of source')
parser.add_argument('--weight_mix', type=float, default=3,  help='Weight of loss of mixture')
parser.add_argument('--transfer', dest='transfer', action='store_true', help='Do transfer learning')
parser.add_argument('--debugging', dest='debugging', action='store_true', help='Whether enter debug mode')
parser.add_argument('--trainset', type=str, default='../Data/0305_train_sample.pth',  help='Which trainset to use')
parser.add_argument('--testset', type=str, default='../Data/0305_test_sample.pth',  help='Which testset to use')
parser.add_argument('--with_silent', dest='with_silent', action='store_true', help='Have silent samples in the data or not')
parser.add_argument('--transfer_model', type=str, default='',  help='Which model to transfer')
parser.add_argument('--remix_ratio', nargs='+', type=float, help='Remix ratio applied to the sources')
parser.add_argument('--baseline', dest='baseline', action='store_true', help='Run baseline model or not')
parser.add_argument('--ratio_on_rep', dest='ratio_on_rep', action='store_true', help='Apply ratio on mask or not')
parser.add_argument('--train_loss', type=str, default='SDR' , help='')
parser.add_argument('--test_loss', type=str, default='SDR' , help='')


args = parser.parse_args()

# if not args.baseline :
#     assert len(args.remix_ratio) == args.n_src

model_label = time.strftime("%m%d_%H%M%S")
print(model_label)

result_path = '../Result/'+ str(model_label)+'.txt'

if not args.debugging:
    file = open(result_path, 'a+')
    for items in vars(args):
        print(items, vars(args)[items]);
        file.write('%s %s\n'%(items, vars(args)[items]));
    file.flush()

if args.trainset == 'MUSDB':

    traindata = MUSDB_data('train', args.n_src, args.with_silent)
    testdata = MUSDB_data('test', args.n_src, args.with_silent)
    train_loader = torch.utils.data.DataLoader(traindata, batch_size = args.batch, shuffle = True, num_workers = 1)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size = args.batch, shuffle = True, num_workers = 1)
#     pass

if args.trainset == 'Slakh':
    traindata = Slakh_data('train', args.n_src)
    testdata = Slakh_data('test', args.n_src)
    train_loader = torch.utils.data.DataLoader(traindata, batch_size = args.batch, shuffle = True, num_workers = 1)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size = args.batch, shuffle = True, num_workers = 1)
    
# ===== Model Define ====
G_model = ConvTasNet(n_src=args.n_src, sample_rate=args.sample_rate).cuda()

if args.transfer_model:
    transfer_model_path = '../Model/' + args.transfer_model + '.pth'
elif args.n_src == 2:   
    transfer_model_path = '../Model/mpa_speclean'
elif args.n_src == 3:
    transfer_model_path = '../Model/tmir_libri3mix'
    
if args.transfer_model:
    G_model.load_state_dict(torch.load(transfer_model_path))
    if not args.debugging:
        print('Loaded model from' + transfer_model_path + '\n')
        file.write('Loaded model from' + transfer_model_path + '\n')
        file.flush()  

optimizer = torch.optim.Adam(G_model.parameters(), lr=args.lr)
# pit_loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx') # loss_func(ets, ori)
mse_loss_func = nn.MSELoss().cuda()

# ===== Start Training ====

best_score = 0

for i in range(args.max_epoch):
    
    loss_epoch = []
    start_time = time.time()
    G_model.train()
    
    for batches, data in enumerate(train_loader):
       
        # mixture -> (bt, L)
        # source -> (bt, n_src, L)
        
        if not isinstance(data, torch.Tensor):
            sources = data[1].to(torch.float32).cuda()
            mixture = data[0].to(torch.float32).cuda()
        else:
            sources = data.to(torch.float32).cuda()
            mixture = torch.sum(sources, 1)

        bt = len(mixture)
        
        mask_ratio = None
        source_ratio = None
        if not args.baseline:
            source_ratio, mask_ratio = utils.sampling_ratio(bt, args.n_src, args.ratio_on_rep)
            remixture = torch.sum(sources * source_ratio, dim=1) # shape - (bt, length)       
        mixture = torch.unsqueeze(mixture, dim=1) # shape (bt, 1, length)
        
        est_sources, masked_tf_rep = G_model(mixture, mask_ratio)
        # est_sources - shape (bt, n_src, wav_length)
        # masked_tf_rep - shape (bt, n_src, 512, feature_length)
        
        sdr_mix_loss = 0
        if not args.baseline:
            if args.ratio_on_rep:
                sources = sources * source_ratio
                est_remixture = torch.sum(est_sources, dim=1)
                
            else:
                est_remixture = torch.sum(est_sources * source_ratio, dim=1) # shape - (bt, length)
#             mse_loss_val = mse_loss_func(est_remixture, remixture)  
            sdr_mix_loss = utils.sdr_score(remixture, est_remixture, f = args.train_loss, cuda=True)

        sdr_src_loss = utils.sdr_score(sources, est_sources, f = args.train_loss, cuda=True)
        
        loss_val = - args.weight_src * sdr_src_loss - args.weight_mix * sdr_mix_loss
        loss_epoch.append(loss_val.cpu().data.numpy())
        
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
#         if batches % 100 == 0:
        
        if args.debugging:
            break
            
    sep_score, remix_score= test_model(G_model, test_loader, args.n_src, args.debugging, args.ratio_on_rep, args.baseline, args.test_loss)

    end_time = time.time()

    result_info = 'Epoch: {} num_batches: {} time: {:.2f} \nsep_score: {:.2f} remix_score: {:.2f} train_loss: {:.2f} best_score: {:.2f}\n'\
                  .format(i, batches, end_time-start_time, sep_score, remix_score, np.mean(loss_epoch), best_score)

    save_score = sep_score if args.baseline else remix_score
#             save_score = sep_score if args.baseline else remix_score_src
    if save_score > best_score:
        if not args.debugging:
            torch.save(G_model.state_dict(), '../Model/'+model_label+'.pth')
        best_score = save_score
        result_info += 'Got best_score. Model saved \n'

    print(result_info)
    if not args.debugging:
        file.write(result_info)
        file.flush()  
    loss_epoch = []
    
#         if args.debugging:
#             break
#     break
                
    # print('Finished Epoch')
       
        