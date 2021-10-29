import os
import time
import glob
import torch
# import librosa
import argparse
import mir_eval
import numpy as np
import soundfile as sf
from utils import *
from tqdm import tqdm
from torch.utils import data
import IPython.display as ipd
from asteroid.models import ConvTasNet
from scipy.linalg import lstsq
# from asteroid.utils.torch_utils import pad_x_to_y
# from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

os.environ['TZ'] = 'EST+05EDT,M4.1.0,M10.5.0'
time.tzset()

def db_to_amp(x):
    return torch.pow(10, torch.true_divide(x,20))

def amp_to_db(x):
#     print('amp', x[:,0])
    return 20*np.log10(x+1e-20)

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

def cal_loudness(stems, mix):
    
    # stems - shape (4, L)
    # mix - shape (L,)
    # stems.dot(scalars) = mix
    p, res, rnk, s = lstsq(stems.T, mix[:, None]) # p.shape (4,)
    p = np.array(p)
    
    return p

def cal_loudness_score(stems, mix, mix_h):
    
    # stems - shape (4, L)
    # mix - shape (L,)
    # stems.dot(scalars) = mix
    
    p = cal_loudness(stems, mix)
    p_h = cal_loudness(stems, mix_h)
#     print('p', p[:,0])
#     print('p_h', p_h[:,0])
    output = sum(abs(amp_to_db((p+1e-20)/(p_h+1e-20))))
#     print(abs(amp_to_db(p/p_h))[:,0])
    
    return output, p, p_h


def eval_model(n_src = 2, model_name1 = '0308_104450' , dataset = None, input_ratio=None, ratio_on_rep = False, model_name2 = '', ratio_on_rep2 = False, path=None, adjust=None, loss_f='SDSDR', with_silent=True):
    
    assert not ratio_on_rep2
    
    #### Load model ####

    G_model = ConvTasNet(n_src=n_src, add_scalar=args.add_scalar, simple=args.simple).cuda()
    G_model.load_state_dict(torch.load('../Model/'+model_name1+'.pth'))
    G_model.eval()  
    
    ### Get data and start evaluation for each song ####
    
#     scores = []
    scores2 = []
    sdr_scores = []
    loud_scores = []
    bss_sdr_list = []
    bss_sir_list = []
    bss_sar_list = []
    ratio_list = []
    
    if with_silent and dataset=='MUSDB':
        folderPath = '../Data/' + dataset + '/large_eval/*'
    else:
        folderPath = '../Data/' + dataset + '/eval/*'
    if n_src ==5:
        folderPath = '../Data/' + dataset + '/eval_5/*'
#     folderPath = '../Data/' + dataset + '/large_eval/*'

    files = glob.glob(folderPath)
    
    input_ratio_list = torch.load('../eval_results/ratios.pt')
    
    #### Load 2nd model
    if model_name2:
        Base_model = ConvTasNet(n_src=n_src, sample_rate=args.sample_rate, add_scalar=args.add_scalar).cuda()
        Base_model.load_state_dict(torch.load('../Model/'+ model_name2 +'.pth'))
        Base_model.eval() 
        
    idx = 0
#     seg_idx = 0

    input_ratio_list = []
    interval = 3 if n_src == 4 else 6
    for a in range(n_src):
        for r in range(-24, 27, interval):
            scale = [0] * n_src
            scale[a] = r
            input_ratio_list.append(scale)
    
    # Generate sample - vocal 12dB, bass 12dB, none 0dB || order: vocal drum bass others
#     input_ratio_list = [[0, 0, 0, 0]]
            
    score_ranges_sdr = [[] for _ in range(len(input_ratio_list))]
    score_ranges_sdsdr = [[] for _ in range(len(input_ratio_list))]
    score_ranges_min = [[] for _ in range(len(input_ratio_list))]

    for f in tqdm(files):
        
#         # only for generate samples
#         f = files[13]
#         print(f)
        
        if idx == 5:
            break
        
        if 'config' in f:
            continue
            
        piece = torch.load(f)[:,:n_src, :] # (N, n_src, L)
        sdr_piece_list = []
        idx += 1
        
        i = len(piece)//30//2*30
        
        sources = piece[i:i+30]
        if len(sources) != 30:
            continue

        mixture = torch.sum(sources, 1).to(torch.float).cuda() # (N, L)

        rec_sources = window_and_recover(sources)
        orig_mix = torch.sum(rec_sources, 0)
        
    #  torch.save(orig_mix,'{}/{}_mixture_{}.pth'.format(path, model_name1, str(idx)))
        
        # ===== calculate performance under different remix scale ====

        p_list = []
        ph_list = []
        for i_ratio, input_ratio in enumerate(input_ratio_list):
#             if i_ratio == 3:
#                 break
            ir = db_to_amp(torch.tensor(input_ratio))
            source_ratio, mask_ratio = sampling_ratio(1, n_src, ratio_on_rep, ir)

            #### 1st model ####
            
            ratio = source_ratio[0].cpu().data # shape-(n_src, 1)
            remixture = torch.sum(rec_sources * ratio, 0)
            
    #   print(sdr_score(remixture[None,:], remixture[None,:], f = loss_f))
    #   print(sdr_score(orig_mix[None,:], remixture[None,:], f = loss_f))

            with torch.no_grad():
                est_sources, masked_tf_rep = G_model(mixture, mask_ratio, source_ratio)  # est_sources - shape (N, n_src, L)     
            rec_est_sources1 = window_and_recover(est_sources) # (n_src, L)

            ratio_list.append(ratio[:,0])
            
            if ratio_on_rep:
                est_remixture1 = torch.sum(rec_est_sources1, 0)
                rec_sources1 = rec_sources * ratio
            else:
                est_remixture1 = torch.sum(rec_est_sources1 * ratio, 0)
                rec_sources1 = rec_sources
            
            mix_sdr = sdr_score(remixture[None,:], est_remixture1[None,:], f = 'SDR')
            mix_sdsdr = sdr_score(remixture[None,:], est_remixture1[None,:], f = 'SDSDR')
            mix_min = min(mix_sdr, mix_sdsdr)
            
            loud_p, p, p_h = cal_loudness_score(rec_sources, remixture, est_remixture1)

            #### record score and save results
    #       sdr_scores.append(mix_sdr)
#             score_ranges_sdr[i_ratio].append(mix_sdr)
#             score_ranges_sdsdr[i_ratio].append(mix_sdsdr)
            score_ranges_min[i_ratio].append(mix_min)
            
            loud_scores.append(abs(loud_p))

#                         print(mix_sdr)

            # ipd.Audio(x, rate=44100)
            # print(x.shape)
            
#             remixture = remixture.cpu().data.numpy()
            
#             if ratio_on_rep:
#                 sf.write('{}/remixture_{}_{}.wav'.format(path, str(idx), str(i_ratio)), remixture/max(abs(remixture)), 44100, 'PCM_24')

#                 orig_mix = orig_mix.cpu().data.numpy()
#                 sf.write('{}/mixture_{}_{}.wav'.format(path, str(idx), str(i_ratio)) , orig_mix/max(abs(orig_mix)), 44100, 'PCM_24')
                     
#             est_remixture1 = est_remixture1.cpu().data.numpy()
#             sf.write('{}/est_remixture_{}_{}.wav'.format(path, str(idx), str(i_ratio)), est_remixture1/max(abs(est_remixture1)), 44100, 'PCM_24')
                     
#             torch.save(remixture, '{}/{}_remixture_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
#             torch.save(orig_mix,'{}/{}_mixture_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
#             torch.save(rec_sources1, '{}/{}_sources_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
#             torch.save(rec_est_sources1, '{}/{}_est_sources_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
#             torch.save(est_remixture1, '{}/{}_est_remixture_{}_{}.pth'.format(path, model_name1, str(idx), str(i_ratio)))
        
#         ==== calculate BSS score for one time ====
            if i_ratio > 0:
                rec_sources1 = rec_sources1.cpu().data.numpy()
                rec_est_sources1 = rec_est_sources1.cpu().data.numpy()
                sdr, isr, sir, sar, perm = mir_eval.separation.bss_eval_images(
                    rec_sources1+1e-20,rec_est_sources1+1e-20, compute_permutation=False)
                
                bss_sdr_list.append(sdr)
                bss_sir_list.append(sir)
                bss_sar_list.append(sar)
                print(sdr)
                print(sir)
                print(sar)

        break
            
    return sdr_scores, loud_scores, bss_sdr_list, bss_sir_list, bss_sar_list, ratio_list, score_ranges_min

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Testing model')

    parser.add_argument('--model_name1', type=str, default='',  help='model_name');
    parser.add_argument('--model_name2', type=str, default='',  help='Dataset');
    parser.add_argument('--dataset', type=str, default='',  help='Dataset');
    parser.add_argument('--with_silent', dest='with_silent', action='store_true', help='whether use silent segment for training')
    parser.add_argument('--ratio_on_rep', dest='ratio_on_rep', action='store_true', help='')
    parser.add_argument('--n_src', type=int, default=2,  help='Number of sources/masks')
    parser.add_argument('--input_ratio', type=list, default=[],  help='source ratio')
    parser.add_argument('--adjust', type=int, default=None,  help='Source to adjust')
    parser.add_argument('--loss_f', type=str, default=None,  help='SDR function to use')
    
    parser.add_argument('--add_scalar', dest='add_scalar', action='store_true', help='whether add scalars')
    parser.add_argument('--simple', dest='simple', action='store_true', help='whether add scalars')
    
    args = parser.parse_args()
    
    time_label = time.strftime("%m%d")
    path = '../eval_results/'+args.model_name1+'_'+args.dataset +'_'+ time_label
    if not isinstance(args.adjust, type(None)):
        path += '_' + str(args.adjust)
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    sdr_scores, loud_scores, bss_sdr_list, bss_sir_list, bss_sar_list, ratios, score_ranges_min = \
    eval_model(n_src = args.n_src, 
               model_name1 = args.model_name1 , 
               dataset = args.dataset, 
               input_ratio=None, 
               ratio_on_rep = args.ratio_on_rep, 
               model_name2 = args.model_name2, 
               path=path, 
               adjust = args.adjust, 
               loss_f = args.loss_f, 
               with_silent=args.with_silent
              )
    # scores -> [15, N]
    # bss_sdr_list -> [N, 4]
#     print('remixture sdr', np.mean(scores), np.std(scores))
#     print('source sdr', np.mean(bss_sdr_list, 0), np.std(bss_sdr_list, 0))
#     print('source sir', np.mean(bss_sir_list, 0), np.std(bss_sir_list, 0))
#     print('source sar', np.mean(bss_sar_list, 0), np.std(bss_sar_list, 0))
    
#     results = {'sdr_scores': sdr_scores, 'loud_scores': loud_scores, 'sdr':bss_sdr_list, 'sir':bss_sir_list, 'sar':bss_sar_list, 'ratios': ratios}
#     scores = eval_model(model, test_loader, args.n_src, False, args.ratio_on_rep, args.baseline, args.ratio_on_rep_mix)
    
#     mean = round(np.mean(sdr_scores), 2)
#     std = round(np.std(sdr_scores), 2)
#     print('sdr_scores', mean, std)

#     print('score_ranges_sdr', np.mean(score_ranges_sdr, 1))
#     print('score_ranges_sdsdr', np.mean(score_ranges_sdsdr, 1))

    
    results = {'min_scores': np.array(score_ranges_min), 
               'loud_scores': loud_scores,
               'bss_sdr_list': bss_sdr_list, 
               'bss_sir_list': bss_sir_list, 
               'bss_sar_list': bss_sar_list,
              }
    
#     torch.save(results, 'samples/{}_min_scores_{}_images.npy'.format(args.model_name1, args.dataset))
    
#     score_ranges_min = np.array(score_ranges_min).reshape(17, 4, -1)
#     score_ranges_min = score_ranges_min.reshape(17, -1)
#     print('score_ranges_min', np.mean(score_ranges_min, 1))
#     print(np.mean(score_ranges_min))
    
    
#     mean = round(np.mean(loud_scores), 2)
#     std = round(np.std(loud_scores), 2)
#     print('loud_scores', mean, std)
    
#     torch.save(results, path+'/results.pt')
    