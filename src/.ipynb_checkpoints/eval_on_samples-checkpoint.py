import os
import time
import glob
import torch
import librosa
import argparse
import numpy as np
from utils import *
from tqdm import tqdm
from torch.utils import data
import IPython.display as ipd
from asteroid.models import ConvTasNet
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

os.environ['TZ'] = 'EST+05EDT,M4.1.0,M10.5.0'
time.tzset()

def db_to_amp(x):
    return torch.pow(10, torch.true_divide(x,20))

def sampling_ratio(bt, n_src, ratio_on_rep, db_ratio = None):
    
    if isinstance(db_ratio, type(None)):
        db_ratio = (torch.rand(bt, n_src)) * DB_FLUC # No decreasing
        amp_ratio = db_to_amp(db_ratio)
    else:
        
        db_ratio = (torch.tensor(db_ratio) + torch.rand(n_src)/5)[None, :]
        amp_ratio = db_ratio
    source_ratio = amp_ratio[:, :, None].cuda() # shape - (None, n_src, None)

    mask_ratio = None
    if ratio_on_rep:
        mask_ratio = amp_ratio[:, :, None, None].cuda() # shape - (None, n_src, None, None) 
    
    return source_ratio, mask_ratio

def window_and_recover(est_sources): # (N, n_src, L)
    
    ### Define window
    seg_length = sr = 44100
    overlap = seg_length//10
    seg_interval = 44100//10*9

    window = np.hanning(overlap*2-1) 
    window = np.concatenate((window[:overlap],np.ones(seg_length-overlap*2),window[overlap-1:]))
    window = window.reshape(1,-1)
    window = window.astype(np.float32)
    
    rec_sources = []
    N = est_sources.shape[0]
    length = (N-1) * seg_interval + seg_length
    ### Recover sources
    n_src = est_sources.shape[1]
    for i in range(n_src):
        wave_src = np.zeros(length)
        src = est_sources[:,i,:].cpu().data.numpy()# (N, 44100)
        for k in range(length//seg_interval):
            wave_src[k*seg_interval:k*seg_interval+seg_length] += src[k] * window[0]
        rec_sources.append(wave_src)
    
    return torch.tensor(rec_sources)

def get_ratio(n_src=4):
    
    num = [0, 10]
    output = []
        
    def add_r(inp):

        if len(inp) >= n_src:
            output.append(inp)
            return
        else:
            for i in num:
                oup = inp + [i]
                add_r(oup)
    add_r([])
    
    return output

def eval_model(n_src = 2, model_name1 = '0308_104450' , dataset = None, input_ratio=None, ratio_on_rep1 = False, model_name2 = '', ratio_on_rep2 = False, path=None):
    
    assert not ratio_on_rep2
    
    #### Load model ####

    G_model = ConvTasNet(n_src=n_src).cuda()
    G_model.load_state_dict(torch.load('../Model/'+model_name1+'.pth'))
    G_model.eval()  
    
    ### Get data and start evaluation for each song ####
    
    scores = []
    scores2 = []
    ratios = []
    
    folderPath = '../Data/' + dataset + '/eval/*'
    files = glob.glob(folderPath)
    
    input_ratio_list = torch.load('../eval_results/ratios.pt')
    
    if model_name2:
                #### Load 2nd model

        Base_model = ConvTasNet(n_src=n_src).cuda()
        Base_model.load_state_dict(torch.load('../Model/'+ model_name2 +'.pth'))
        Base_model.eval() 
        
    idx = 0

    for f in tqdm(files):
#         print(files)
        
#         input_ratio_list = get_ratio(n_src = n_src)
# #         print(input_ratio_list)
# #         fake()
        input_ratio_list = [[10, -10,-10,-10]]
        if idx == 50:
            break
        
        if 'config' not in f:
            sources = torch.load(f)[150:180,:n_src, :] # (N, n_src, L)
            if len(sources) != 30:
                continue
#             print(len(sources))
            mixture = torch.sum(sources, 1).to(torch.float).cuda() # (N, L)

#             bt = mixture.shape[0]
                
#             input_ratio = input_ratio_list[idx]
            idx += 1
#             print(input_ratio)
            for i_ratio, input_ratio in enumerate(input_ratio_list):
#             for i in range(1):
                ir = db_to_amp(torch.tensor(input_ratio))
#                 print(ir)

                source_ratio, mask_ratio = sampling_ratio(1, n_src, ratio_on_rep1, ir)
#                 print(mask_ratio)
                #### 1st model ####
                with torch.no_grad():

                    ratios.append(source_ratio[0,:,0])
                    print(source_ratio[0,:,0])
                    est_sources, masked_tf_rep = G_model(mixture, mask_ratio)
                    # est_sources - shape (N, n_src, L)

                rec_sources = window_and_recover(sources)
                rec_est_sources1 = window_and_recover(est_sources) # (n_src, L)

                ratio = source_ratio[0].cpu().data # shape-(n_src, 1)
                remixture = torch.sum(rec_sources * ratio, 0)
                if ratio_on_rep1:
                    est_remixture1 = torch.sum(rec_est_sources1, 0)
                    rec_sources1 = rec_sources * ratio
                else:
                    est_remixture1 = torch.sum(rec_est_sources1 * ratio, 0)
                    rec_sources1 = rec_sources

                mix_sdr = SDR(remixture, est_remixture1)

                #### record score and save results

                scores.append(mix_sdr)  
                print(mix_sdr)
                torch.save(remixture, '{}/{}_remixture_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
                torch.save(rec_sources1, '{}/{}_sources_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
                torch.save(rec_est_sources1, '{}/{}_est_sources_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
                torch.save(est_remixture1, '{}/{}_est_remixture_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))

                if model_name2:

                    ###### Eval 2nd model ####
                    with torch.no_grad():

                        est_sources2, masked_tf_rep2 = Base_model(mixture)
                        # est_sources - shape (N, n_src, L)

                    rec_est_sources2 = window_and_recover(est_sources2) # (n_src, L)

                    est_remixture2 = torch.sum(rec_est_sources2 * ratio, 0)

                    mix_sdr2 = SDR(remixture, est_remixture2)

                    ### Record score ####

                    scores2.append(mix_sdr2)
                    print(mix_sdr2)
                    
                    torch.save(rec_sources, '{}/{}_sources_{}_{}.pth'.format(path, model_name2, str(idx), str(i_ratio)))
                    torch.save(rec_est_sources2, '{}/{}_est_sources_{}_{}.pth'.format(path, model_name2, str(idx), str(i_ratio)))
                    torch.save(est_remixture2, '{}/{}_est_remixture_{}_{}.pth'.format(path, model_name2, str(idx), str(i_ratio)))
#                     idx += 1
                               
#         break

#         break
            
    return scores, scores2, ratios

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Testing model')

    parser.add_argument('--model_name1', type=str, default='',  help='model_name');
    parser.add_argument('--model_name2', type=str, default='',  help='Dataset');
    parser.add_argument('--dataset', type=str, default='',  help='Dataset');
    parser.add_argument('--ratio_on_rep1', dest='ratio_on_rep1', action='store_true', help='')
    parser.add_argument('--n_src', type=int, default=2,  help='Number of sources/masks')
    parser.add_argument('--input_ratio', type=list, default=[],  help='source ratio')
    
    args = parser.parse_args()
    
    time_label = time.strftime("%m%d")
    path = '../eval_results/'+args.model_name1+'_'+args.dataset +'_'+ time_label
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    scores, scores2, ratios = eval_model(n_src = args.n_src, model_name1 = args.model_name1 , dataset = args.dataset, input_ratio=None, ratio_on_rep1 = args.ratio_on_rep1, model_name2 = args.model_name2, path=path)
    
    print(np.mean(scores), np.mean(scores2))
    
    results = {'scores': scores, 'scores2': scores2, 'ratios': ratios}
#     scores = eval_model(model, test_loader, args.n_src, False, args.ratio_on_rep, args.baseline, args.ratio_on_rep_mix)
    
    torch.save(results, path+'/results.pt')
    