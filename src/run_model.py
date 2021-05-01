
from utils import *
def run_model(model, mixture, n_src, baseline, ratio_on_rep, weight_src = 0, weight_mix = 0)

    model_label = time.strftime("%m%d_%H%M%S")
    mask_ratio = None
    if not baseline:
        source_ratio, mask_ratio = sampling_ratio(bt, n_src, ratio_on_rep)
        
    remixture = torch.sum(sources * source_ratio, dim=1)

    mixture = torch.unsqueeze(mixture, dim=1) # shape (bt, 1, length)
    est_sources, masked_tf_rep = model(mixture, mask_ratio)

    # est_sources - shape (bt, n_src, wav_length)
    # masked_tf_rep - shape (bt, n_src, 512, feature_length)

    sdr_mix_loss = 0
    if ratio_on_rep:

        sources = sources * source_ratio
        est_remixture_src = torch.sum(est_sources, dim=1)
     
    else:
        est_remixture_src = torch.sum(est_sources * source_ratio, dim=1)

    sdr_mix_loss = SDR(remixture, est_remixture_src, cuda=True)  
    # sdr_mix_loss = SDR(remixture, est_remixture_mix, cuda=True)
    sdr_src_loss = SDR(sources, est_sources, cuda=True)

    loss_val = - weight_src * sdr_src_loss - weight_mix * sdr_mix_loss

    return loss_val, sdr_src_loss, sdr_mix_loss, 0