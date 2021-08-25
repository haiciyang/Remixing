import os
import time
import glob
import torch
# import librosa
import argparse
import mir_eval
import numpy as np
from utils import *
from tqdm import tqdm
from torch.utils import data
import IPython.display as ipd
from asteroid.models import ConvTasNet
# from asteroid.utils.torch_utils import pad_x_to_y
# from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

os.environ['TZ'] = 'EST+05EDT,M4.1.0,M10.5.0'
time.tzset()

def db_to_amp(x):
    return torch.pow(10, torch.true_divide(x,20))

def sampling_ratio(bt, n_src, ratio_on_rep, db_ratio = None):
    
    if isinstance(db_ratio, type(None)):
        db_ratio = (torch.rand(bt, n_src)) * DB_FLUC # No decreasing
        amp_ratio = db_to_amp(db_ratio)
    else:
#         db_ratio = (db_ratio + torch.rand(n_src)/5)[None, :]
        db_ratio = db_ratio[None, :]
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

def eval_model(n_src = 2, model_name1 = '0308_104450' , dataset = None, input_ratio=None, ratio_on_rep1 = False, model_name2 = '', ratio_on_rep2 = False, path=None, adjust=None):
    
    assert not ratio_on_rep2
    
    #### Load model ####

    G_model = ConvTasNet(n_src=n_src).cuda()
    G_model.load_state_dict(torch.load('../Model/'+model_name1+'.pth'))
    G_model.eval()  
    
    ### Get data and start evaluation for each song ####
    
#     scores = []
    scores2 = []
    scores = []
    bss_sdr_list = []
    bss_sir_list = []
    bss_sar_list = []
    ratio_list = []
    
    folderPath = '../Data/' + dataset + '/eval/*'
    if n_src ==5:
        folderPath = '../Data/' + dataset + '/eval_5/*'
    files = glob.glob(folderPath)
    
    input_ratio_list = torch.load('../eval_results/ratios.pt')
    
    #### Load 2nd model
    if model_name2:
        Base_model = ConvTasNet(n_src=n_src).cuda()
        Base_model.load_state_dict(torch.load('../Model/'+ model_name2 +'.pth'))
        Base_model.eval() 
        
    idx = 0
#     seg_idx = 0

    for f in tqdm(files):
        
#         if not isinstance(adjust, type(None)):
            
#             input_ratio_list = []
#             for r in range(-24, 27, 3):
#                 scale = [0,0,0,0]
#                 scale[adjust] = r
#                 input_ratio_list.append(scale)
#         else:
#             input_ratio_all = []
#             interval = 3 if n_src == 4 else 6
#             for a in range(n_src):
#                 for r in range(-24, 27, interval):
#                     scale = [0] * n_src
#                     scale[a] = r
#                     input_ratio_all.append(scale)
                
#             if n_src == 2:
#                 input_ratio_list = [[12, 0],[0, 12],[0, 0],[12, 12], [12, -12],[-12, 12]]
#             if n_src == 3:
#                 input_ratio_list = [[12, 0,0],[0, 12,0],[0, 0,12],[0, 0,0],[12, 12, 0], [12, 0, 12], [0, 12, 12], [12, -12,-12],[-12, 12,-12],[-12, -12,12]]
#             if n_src == 4:
#                 input_ratio_list = [[12, 0,0,0],[0, 12,0,0],[0, 0,12,0],[0, 0,0,12],[0, 0,0,0],[12, 12, 0, 0], [12, 0, 12, 0],  [12, 0, 0, 12], [0, 12, 12, 0], [0, 12, 0, 12],  [0, 0, 12, 12], [12, -12,-12,-12],[-12, 12,-12,-12],[-12, -12,12,-12],[-12, -12,-12,12]]
#             if n_src == 5:
#                 input_ratio_list = [[12, 0,0,0,0],[0, 12,0,0,0],[0, 0,12,0,0],[0, 0,0,12,0],[0, 0,0,0,12],[0, 0,0,0,0],[12, 12, 0, 0,0], [12, 0, 12, 0,0],  [12, 0, 0, 12,0], [12, 0, 0, 0, 12,],[0, 12, 12, 0, 0], [0, 12, 0, 12, 0], [0, 12, 0, 0,12], [0, 0, 12, 12, 0], [0, 0, 12,  0, 12],[0, 0, 0,12, 12],[12, -12,-12,-12, -12],[-12, 12,-12,-12, -12],[-12, -12,12,-12, -12],[-12, -12,-12,12, -12],[-12, -12,-12, -12,12]]
        
        if idx == 11:
            break
        
        input_ratio_list = [[12, 0,0,0],[0, 12,0,0],[0, 0,12,0],[0, 0,0,12], [0,0,0,0]]

#         input_ratio_list = [[0,0,0,0]]
        
        if 'config' not in f:
            
            piece = torch.load(f)[:,:n_src, :] # (N, n_src, L)
            sdr_piece_list = []
            idx += 1
            
            i = len(piece)//30//2*30
            for l in range(1):
                sources = piece[i:i+30]
                if len(sources) != 30:
                    continue
                    
#                 if not isinstance(adjust, type(None)): ## For the ratio specific task, only use one segment from each songs
# #                     print(i)
# #                     print(len(piece)//30//2*30)
#                     if i != len(piece)//30//2*30:
#                         continue
# #                     fake()
        
                mixture = torch.sum(sources, 1).to(torch.float).cuda() # (N, L)

#                 seg_idx += 1
#                 print('song', song_idx, 'seg', seg_idx) 
                
#                 if isinstance(adjust, type(None)): # Not running every ratio for every segment
#                     input_ratio_list = [input_ratio_all[seg_idx % len(input_ratio_all)]]

                if not ratio_on_rep1:
                    with torch.no_grad():
                        est_sources, masked_tf_rep = G_model(mixture, None)    
                    
                rec_sources = window_and_recover(sources)
                orig_mix = torch.sum(rec_sources, 0)
                
                torch.save(orig_mix,'{}/{}_mixture_{}.pth'.format(path, model_name1, str(idx)))
                
                # ===== calculate performance under different remix scale ====
                for i_ratio, input_ratio in enumerate(input_ratio_list):
                    ir = db_to_amp(torch.tensor(input_ratio))
                    source_ratio, mask_ratio = sampling_ratio(1, n_src, ratio_on_rep1, ir)

                    #### 1st model ####

                    if ratio_on_rep1:
                        with torch.no_grad():
                            est_sources, masked_tf_rep = G_model(mixture, mask_ratio)
                            # est_sources - shape (N, n_src, L)
                    
                    
                    rec_est_sources1 = window_and_recover(est_sources) # (n_src, L)

                    ratio = source_ratio[0].cpu().data # shape-(n_src, 1)
#                     print(ratio[:,0])
                    ratio_list.append(ratio[:,0])
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
                    
                    

#                     i_ratio = 20
#                     print(mix_sdr)
                    torch.save(remixture, '{}/{}_remixture_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
#                     torch.save(orig_mix,'{}/{}_mixture_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
    #                 torch.save(rec_sources1, '{}/{}_sources_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
#                     torch.save(rec_est_sources1, '{}/{}_est_sources_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
                    torch.save(est_remixture1, '{}/{}_est_remixture_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
                
                # ==== calculate BSS score for one time ====
#                 if i_ratio == 0:
#                     rec_sources1 = rec_sources1.cpu().data.numpy()
#                     rec_est_sources1 = rec_est_sources1.cpu().data.numpy()
#                     sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
#                         rec_sources1+1e-20,rec_est_sources1+1e-20, compute_permutation=False)
#                     bss_sdr_list.append(sdr)
#                     bss_sir_list.append(sir)
#                     bss_sar_list.append(sar)


#                 if model_name2:

#                     ###### Eval 2nd model ####
#                     with torch.no_grad():

#                         est_sources2, masked_tf_rep2 = Base_model(mixture)
#                         # est_sources - shape (N, n_src, L)

#                     rec_est_sources2 = window_and_recover(est_sources2) # (n_src, L)

#                     est_remixture2 = torch.sum(rec_est_sources2 * ratio, 0)

#                     mix_sdr2 = SDR(remixture, est_remixture2)

#                     ### Record score ####

#                     scores2.append(mix_sdr2)
#                     print(mix_sdr2)
                    
#                     torch.save(rec_sources, '{}/{}_sources_{}_{}.pth'.format(path, model_name2, str(idx), str(i_ratio)))
#                     torch.save(rec_est_sources2, '{}/{}_est_sources_{}_{}.pth'.format(path, model_name2, str(idx), str(i_ratio)))
#                     torch.save(est_remixture2, '{}/{}_est_remixture_{}_{}.pth'.format(path, model_name2, str(idx), str(i_ratio)))
#                     idx += 1
                               
#         break

#         break
            
    return scores, bss_sdr_list, bss_sir_list, bss_sar_list, ratio_list

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Testing model')

    parser.add_argument('--model_name1', type=str, default='',  help='model_name');
    parser.add_argument('--model_name2', type=str, default='',  help='Dataset');
    parser.add_argument('--dataset', type=str, default='',  help='Dataset');
    parser.add_argument('--ratio_on_rep1', dest='ratio_on_rep1', action='store_true', help='')
    parser.add_argument('--n_src', type=int, default=2,  help='Number of sources/masks')
    parser.add_argument('--input_ratio', type=list, default=[],  help='source ratio')
    parser.add_argument('--adjust', type=int, default=None,  help='Source to adjust')
    
    args = parser.parse_args()
    
    time_label = time.strftime("%m%d")
    path = '../eval_results/'+args.model_name1+'_'+args.dataset +'_'+ time_label
    if not isinstance(args.adjust, type(None)):
        path += '_' + str(args.adjust)
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    scores, bss_sdr_list, bss_sir_list, bss_sar_list, ratios = eval_model(n_src = args.n_src, model_name1 = args.model_name1 , dataset = args.dataset, input_ratio=None, ratio_on_rep1 = args.ratio_on_rep1, model_name2 = args.model_name2, path=path, adjust = args.adjust)
    # scores -> [15, N]
    # bss_sdr_list -> [N, 4]
#     print('remixture sdr', np.mean(scores), np.std(scores))
#     print('source sdr', np.mean(bss_sdr_list, 0), np.std(bss_sdr_list, 0))
#     print('source sir', np.mean(bss_sir_list, 0), np.std(bss_sir_list, 0))
#     print('source sar', np.mean(bss_sar_list, 0), np.std(bss_sar_list, 0))
    
    results = {'scores': scores, 'sdr':bss_sdr_list, 'sir':bss_sir_list, 'sar':bss_sar_list, 'ratios': ratios}
#     scores = eval_model(model, test_loader, args.n_src, False, args.ratio_on_rep, args.baseline, args.ratio_on_rep_mix)
    
    torch.save(results, path+'/results.pt')
    