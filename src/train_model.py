import os
import time
import torch
import librosa
import argparse
import numpy as np
import torch.nn as nn
import IPython.display as ipd
from matplotlib import pyplot as plt
from asteroid.models import ConvTasNet
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


from utils import *

os.environ['TZ'] = 'EST+05EDT,M4.1.0,M10.5.0'
time.tzset()


parser = argparse.ArgumentParser(description='train_model')

parser.add_argument('--lr', type=float, default=0.001,  help='Number of sources')
parser.add_argument('--max_epoch', type=int, default=100,  help='Input length to the network')
parser.add_argument('--n_src', type=int, default=2,  help='Number of sources/masks')
parser.add_argument('--sample_rate', type=int, default=44100,  help='Sampling rate')
parser.add_argument('--weight_src', type=float, default=1,  help='Weight of loss of source')
parser.add_argument('--weight_mix', type=float, default=3,  help='Weight of loss of mixture')
parser.add_argument('--transfer', dest='transfer', action='store_true', help='Do transfer learning')
parser.add_argument('--debugging', dest='debugging', action='store_true', help='Whether enter debug mode')
parser.add_argument('--trainset', type=str, default='../Data/0305_train_sample.pth',  help='Which trainset to use')
parser.add_argument('--testset', type=str, default='../Data/0305_test_sample.pth',  help='Which testset to use')
parser.add_argument('--transfer_model', type=str, default='',  help='Which model to transfer')
parser.add_argument('--remix_ratio', nargs='+', type=float, help='Remix ratio applied to the sources')
parser.add_argument('--baseline', dest='baseline', action='store_true', help='Run baseline model or not')
parser.add_argument('--ratio_on_rep', dest='ratio_on_rep', action='store_true', help='Apply ratio on mask or not')


# parser.add_argument('--batch', type=int, default=2,  help='Input length to the network');
# parser.add_argument('--dataset', type=str, default='test', help='dataset to process')
# parser.add_argument('--size', type=int, default=100, help='dataset to process')
# parser.add_argument('--save_name', type=str, default='',  help='Save the output data');


args = parser.parse_args()

if not args.baseline:
    assert len(args.remix_ratio) == args.n_src

model_label = time.strftime("%m%d_%H%M%S")
print(model_label)

result_path = '../Result/'+ str(model_label)+'.txt'

if not args.debugging:
    file = open(result_path, 'a+')
    for items in vars(args):
        print(items, vars(args)[items]);
        file.write('%s %s\n'%(items, vars(args)[items]));
    file.flush()

else:
    trainset = '../Data/0305_train_sample.pth'
    testset = '../Data/0305_test_sample.pth'
    
train_loader = torch.load(args.trainset)
test_loader = torch.load(args.testset)

# ===== Model Define ====
G_model = ConvTasNet(n_src=args.n_src, sample_rate=args.sample_rate).cuda()
if args.transfer_model:
    transfer_model_path = '../Model/' + args.transfer_model + '.pth'
elif args.n_src == 2:   
    transfer_model_path = '../Model/mpa_speclean'
elif args.n_src == 3:
    transfer_model_path = '../Model/tmir_libri3mix'
    
if args.transfer:
    G_model.load_state_dict(torch.load(transfer_model_path)['state_dict'])
    if not args.debugging:
        print('Loaded model from' + transfer_model_path + '\n')
        file.write('Loaded model from' + transfer_model_path + '\n')
        file.flush()  

optimizer = torch.optim.Adam(G_model.parameters(), lr=args.lr)
# pit_loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx') # loss_func(ets, ori)
mse_loss_func = nn.MSELoss().cuda()

# ===== Start Training ====

best_score = 0

source_ratio = torch.tensor(args.remix_ratio)[None, :, None].cuda() # shape - (None, n_src, None)
mask_ratio = torch.tensor(args.remix_ratio)[None, :, None, None].cuda() # shape - (None, n_src, None, None) Normalized version

for i in range(args.max_epoch):
    
    loss_epoch = []
    start_time = time.time()
#     test_loss = []
    G_model.train()
#     G_model.eval()
    
    for mixture, sources in train_loader:
        
        # mixture -> (bt, L)
        # source -> (bt, n_src, L)
        
        mixture = mixture.cuda()
        sources = sources.cuda()
        if not args.baseline:
            remixture = torch.sum(sources * source_ratio, dim=1) # shape - (bt, length)
        
        mixture = torch.unsqueeze(mixture, dim=1) # shape (bt, 1, length)
        est_sources, masked_tf_rep = G_model(mixture)
#         print(max(est_sources[0][0]), max(est_sources[0][1]),max(est_sources[0][2]))
#         print(max(sources[0][0]), max(sources[0][1]),max(sources[0][2]))
        # est_sources - shape (bt, n_src, wav_length)
        # masked_tf_rep - shape (bt, n_src, 512, feature_length)

#         pit_loss_val, reordered_est_sources, batch_indices = pit_loss_func(est_sources, sources, return_est=True) 
        sdr_src_loss = SDR(sources, est_sources, cuda=True)
#         print('train_loss', sdr_src_loss)
#         print(pit_loss_val)
#         print(max(reordered_est_sources[0][0]), max(reordered_est_sources[0][1]),max(reordered_est_sources[0][2]))
        
        sdr_mix_loss = 0
        if not args.baseline:
            if args.ratio_on_rep:
#                 reordered_masked_rep = torch.stack(
#                     [torch.index_select(s, 0, b) for s, b in zip(masked_tf_rep, batch_indices)]
#                 ) # (bt, n_src, 512, feature_length)

                masked_mixture = torch.unsqueeze(
                    torch.sum(masked_tf_rep * mask_ratio, dim=1), dim=1
                ) #(bt, 1, 512, feature_length)
                est_remixture = torch.squeeze(G_model.decoder_mix(masked_mixture)) # shape (bt, wav_length)
                est_remixture = pad_x_to_y(est_remixture, remixture)
            else:
                est_remixture = torch.sum(est_sources * source_ratio, dim=1) # shape - (bt, length)
#             mse_loss_val = mse_loss_func(est_remixture, remixture)       
            sdr_mix_loss = SDR(remixture, est_remixture, cuda=True)  
            
        
#         print(pit_loss_val, mse_loss_val)
        loss_val = - args.weight_src * sdr_src_loss - args.weight_mix * sdr_mix_loss
#         loss_epoch.append(loss_val.cpu().data.numpy())
        
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
        if args.debugging:
            break
#     break
            
    sep_score, remix_score = test_model(G_model, test_loader, args.debugging, args.remix_ratio, args.ratio_on_rep, args.baseline)
    
    end_time = time.time()
    
    result_info = 'Epoch: {} time: {:.2f} sep_score: {:.2f} remix_score: {:.2f} \n'\
          .format(i, end_time-start_time, sep_score, remix_score)
    
    save_score = sep_score if args.baseline else remix_score
    if save_score > best_score:
        if not args.debugging:
            torch.save(G_model.state_dict(), '../Model/'+model_label+'.pth')
        best_score = save_score
        result_info += 'Got best_score. Model saved \n'
    
    print(result_info)
    if not args.debugging:
        file.write(result_info)
        file.flush()  
        
    
    
 

