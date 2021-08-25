import time
import torch
import librosa
import mir_eval
import argparse
import numpy as np
from utils import *
from tqdm import tqdm
import IPython.display as ipd
from matplotlib import pyplot as plt
from asteroid.models import ConvTasNet
from generate_MUSdata import MUSDB_data
from generate_SlakhData import Slakh_data
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

def test_model(model, data_loader, n_src, debugging, ratio_on_rep, baseline, ratio_on_rep_mix, loss):
    
#     source_ratio = remix_ratio[None, :, None].cuda() # shape - (None, n_src, None)
#     mask_ratio = remix_ratio[None, :, None, None].cuda() # shape - (None, n_src, None, None) Normalized version
    MAX_APM = 3.162
    FLUC_RANGE = MAX_APM - 1/MAX_APM

    model.eval()
    sep_score = []
    remix_score_mix = []
    remix_score_src = []
#     print(kwargs)

    with torch.no_grad():
        
        epoch = 1 #0 if args.dataset == 'MUSDB' else 1
        
        for ep in range(epoch):
        
            for data in data_loader:

                if not isinstance(data, torch.Tensor):
                    sources = data[1].to(torch.float32).cuda()
                    mixture = data[0].to(torch.float32).cuda()
                else:
    #                 print(data.shape)
                    sources = data.to(torch.float32).cuda()
                    mixture = torch.sum(sources, 1)

                bt = len(mixture)

                mask_ratio = None
    #             if not baseline:
                source_ratio, mask_ratio = sampling_ratio(bt, n_src, ratio_on_rep)
                remixture = torch.sum(sources * source_ratio, dim=1) # shape - (bt, length)

                mixture = torch.unsqueeze(mixture, dim=1)
                est_sources, masked_tf_rep = model(mixture, mask_ratio)

    #             if not baseline:
                if ratio_on_rep:             
                    sources = sources * source_ratio

                    est_remixture_src = torch.sum(est_sources, dim=1)
                    remix_score_src.append(sdr_score(remixture, est_remixture_src, f = loss_f))

                elif ratio_on_rep_mix:

                    _, mask_ratio = sampling_ratio(bt, n_src, True) # generate mask_ratio
                    masked_mixture = torch.unsqueeze(
                        torch.sum(masked_tf_rep * mask_ratio, dim=1), dim=1
                    )

                    est_remixture = model.forward_decoder(masked_mixture)
                    est_remixture_mix = pad_x_to_y(est_remixture, remixture)
                    est_remixture_src = torch.sum(est_sources * source_ratio, dim=1)

                    remix_score_src.append(sdr_score(remixture, est_remixture_src, f = loss_f))
                    remix_score_mix.append(sdr_score(remixture, est_remixture_mix, f = loss_f))

                else:
                    est_remixture = torch.sum(est_sources * source_ratio, dim=1)
                    remix_score_src.append(sdr_score(remixture, est_remixture, f = loss_f, cuda=False)) 

                sep_score.append(sdr_score(sources, est_sources, f = loss_f, cuda=False))

                if debugging:
                    break
            
    remix_score_src = 0 if not remix_score_src else remix_score_src
    remix_score_mix = 0 if not remix_score_mix else remix_score_mix
    
    return np.mean(sep_score), np.mean(remix_score_src), np.mean(remix_score_mix)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Testing model')

    parser.add_argument('--model_name', type=str, default='',  help='model_name');
    parser.add_argument('--dataset', type=str, default='',  help='Dataset');
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='Run baseline model or not')
    parser.add_argument('--ratio_on_rep', dest='ratio_on_rep', action='store_true', help='')
    parser.add_argument('--ratio_on_rep_mix', dest='ratio_on_rep_mix', action='store_true', help='')
    parser.add_argument('--n_src', type=int, default=2,  help='Number of sources/masks')
    
    args = parser.parse_args()
    
    print('hh')
    
    model = ConvTasNet(n_src=args.n_src).cuda()
    model.load_state_dict(torch.load('../Model/'+args.model_name+'.pth'))
    model.eval()
    
    if args.dataset == 'MUSDB':

        testdata = MUSDB_data('test', args.n_src)
        test_loader = torch.utils.data.DataLoader(testdata, batch_size = 2, shuffle = True, num_workers = 1)
    #     pass

    if args.dataset == 'Slakh':
        testdata = Slakh_data('test', args.n_src)
        test_loader = torch.utils.data.DataLoader(testdata, batch_size = 2, shuffle = True, num_workers = 1)
    

    scores = test_model(model, test_loader, args.n_src, False, args.ratio_on_rep, args.baseline, args.ratio_on_rep_mix)
    
    print(scores)