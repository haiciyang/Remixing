import librosa
import torch
import numpy as np
import IPython.display as ipd
from matplotlib import pyplot as plt
from asteroid.models import ConvTasNet
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
import time

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

def test_model(model, data_loader, debugging, remix_ratio, ratio_on_rep, baseline):
    
    source_ratio = torch.tensor(remix_ratio)[None, :, None].cuda() # shape - (None, n_src, None)
    mask_ratio = torch.tensor(remix_ratio)[None, :, None, None].cuda() # shape - (None, n_src, None, None) Normalized version

    model.eval()
    sep_score = []
    remix_score = []
#     print(kwargs)

    with torch.no_grad():
        for mixture, sources in data_loader:

            mixture = mixture.cuda()
            sources = sources.cuda()
            if not baseline:
                remixture = torch.sum(sources * source_ratio, dim=1) # shape - (bt, length)
#             source_0 = torch.unsqueeze(source[0], dim=1)
#             source_1 = torch.unsqueeze(source[1], dim=1)
            
            mixture = torch.unsqueeze(mixture, dim=1)
            est_sources, masked_tf_rep = model(mixture)
#             print(torch.max(sources[0]))
#             print(torch.max(est_sources[0]))
#             pit_loss_val, reordered_est_sources, batch_indices = \
#             pit_loss_func(est_sources, sources, return_est=True) 
#             print(SDR(sources, est_sources, cuda=False))
            sep_score.append(SDR(sources, est_sources, cuda=False))
            
            
            if not baseline:
                if ratio_on_rep:
#                     reordered_masked_rep = torch.stack(
#                         [torch.index_select(s, 0, b) for s, b in zip(masked_tf_rep, batch_indices)]
#                     ) # (bt, n_src, 512, feature_length)

                    masked_mixture = torch.unsqueeze(
                        torch.sum(masked_tf_rep * mask_ratio, dim=1), dim=1
                    ) #(bt, 1, 512, feature_length)

                    est_remixture = torch.squeeze(model.decoder_mix(masked_mixture)) # shape (bt, wav_length)
                    est_remixture = pad_x_to_y(est_remixture, remixture)
                else:
                    est_remixture = torch.sum(est_sources * source_ratio, dim=1)
                
                remix_score.append(SDR(remixture, est_remixture, cuda=False)) 
            
            if debugging:
                break
        
    return np.mean(sep_score), np.mean(remix_score)
