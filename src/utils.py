import time
import torch
import librosa
import mir_eval
import numpy as np
import IPython.display as ipd
from matplotlib import pyplot as plt
from asteroid.models import ConvTasNet
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

DB_FLUC = 12

def SISDR(s,sr,  cuda = False):
    
    eps = 1e-20
    scale = torch.sum(sr * s, dim = 1) / torch.sum(s**2, dim = 1) 
    scale = scale.unsqueeze(dim = 1) # shape - [50,1]
    s = s * scale
    sisdr = torch.mean(10*torch.log10(torch.sum(s**2, dim = 1)/(torch.sum((s-sr)**2, dim=1)+eps)+eps))
    if cuda:
        return torch.mean(sisdr)
    else:
        return torch.mean(sisdr).cpu().data.numpy()
    
def SDR(s,sr,  cuda = False):
    # Input s, sr - (bt, n_src, L)
    eps = 1e-20
    
    inter = 10*torch.log10(torch.sum(s**2, dim = -1)/(torch.sum((s-sr)**2, dim=-1)+eps)+eps)
    sdr = torch.mean(inter)
    
#     sdr = torch.mean(10*torch.log10(torch.sum(s**2, dim = -1)/(torch.sum((s-sr)**2, dim=-1)+eps)+eps))
    
    if cuda:
        return sdr
    else:
        return sdr.cpu().data.numpy()

    
def get_mir_scores(s, n, x, sr):
    """
    s: Source signal
    n: Noise signal
    x: s + n
    sr: Reconstructed signal (or some signal to evaluate against)
    """
    ml = np.int(np.minimum(len(s), len(sr)))
    source = np.array(s[:ml])[:,None].T
    noise = np.array(n[:ml])[:,None].T
    sourceR = np.array(sr[:ml])[:,None].T
    noiseR = np.array(x[:ml]-sr[:ml])[:,None].T
    sdr,sir,sar,_=mir_eval.separation.bss_eval_sources(
            np.concatenate((source, noise),0),
            np.concatenate((sourceR, noiseR),0), 
            compute_permutation=False)   
    # Take the first element from list for source's performance
    return sdr[0],sir[0],sar[0]


def db_to_amp(x):
    return torch.pow(10, torch.true_divide(x,20))

def sampling_ratio(bt, n_src, ratio_on_rep, db_ratio = None):
    
    if isinstance(db_ratio, type(None)):
        db_ratio = (torch.rand(bt, n_src)-0.5) * DB_FLUC * 2
        amp_ratio = db_to_amp(db_ratio)
    else:
        db_ratio = torch.tensor(db_ratio)[None, :]
        amp_ratio = db_ratio
    source_ratio = amp_ratio[:, :, None].cuda() # shape - (None, n_src, None)

    mask_ratio = None
    if ratio_on_rep:
        mask_ratio = amp_ratio[:, :, None, None].cuda() # shape - (None, n_src, None, None) 
    
    return source_ratio, mask_ratio

def test_model(model, data_loader, n_src, debugging, ratio_on_rep, baseline, ratio_on_rep_mix):
    
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
        
        epoch = 3
        
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

                elif ratio_on_rep_mix:

                    _, mask_ratio = sampling_ratio(bt, n_src, True) # generate mask_ratio
                    masked_mixture = torch.unsqueeze(
                        torch.sum(masked_tf_rep * mask_ratio, dim=1), dim=1
                    )

                    est_remixture = model.forward_decoder(masked_mixture)
                    est_remixture_mix = pad_x_to_y(est_remixture, remixture)
                    est_remixture_src = torch.sum(est_sources * source_ratio, dim=1)

                    remix_score_mix.append(SDR(remixture, est_remixture_mix))

                else:
                    est_remixture_src = torch.sum(est_sources * source_ratio, dim=1)
                    
                remix_score_src.append(SDR(remixture, est_remixture_src, cuda=False)) 
                sep_score.append(SDR(sources, est_sources, cuda=False))

                if debugging:
                    break
            
    remix_score_src = 0 if not remix_score_src else remix_score_src
    remix_score_mix = 0 if not remix_score_mix else remix_score_mix
    
    return np.mean(sep_score), np.mean(remix_score_src), np.mean(remix_score_mix)