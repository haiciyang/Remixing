import torch
import librosa
import numpy as np
import IPython.display as ipd
from asteroid.models import ConvTasNet
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

def eval_wave(n_src = 2, model_name = '0308_104450' , num_secs = 60, track = '/media/sdc1/slakh2100_wav/train/Track00001', ratio = [], prop2=False):

    # model_name = '0308_234125'  # 3src

    # test_loader = torch.load('../Data/0308_test_'+str(n_src)+'src.pth')
    G_model = ConvTasNet(n_src=n_src).cuda()
    G_model.load_state_dict(torch.load('../Model/'+model_name+'.pth'))
    G_model.eval()

    mixtures = []
    sources = []

    error_num = 0

    piano = True 
    drum = True if n_src >=2 else False
    bass = True if n_src >=3 else False
    guitar = True if n_src ==4 else False

    piano_id = ''
    drum_id = ''
    bass_id = ''
    guitar_id = ''
    ids = [piano_id, drum_id , bass_id , guitar_id] 
    path = [''] * n_src 
    #             print(track)

    with open(track+'/metadata.yaml') as f:

        lines = f.readlines()

        for i, line in enumerate(lines):
            if piano and 'Piano' in line and not piano_id:
                piano_id = lines[i-2][-5:-2]
                path[0] = track+'/stems/'+piano_id+'.wav'
            if drum and 'Drum' in line and not drum_id:
                drum_id = lines[i-2][-5:-2]
                path[1] = track+'/stems/'+drum_id+'.wav'
            if bass and 'Bass' in line and not bass_id:
                bass_id = lines[i-2][-5:-2]
                path[2] = track+'/stems/'+bass_id+'.wav'
            if guitar and 'Guitar' in line and not guitar_id:
                guitar_id = lines[i-2][-5:-2]
                path[3] = track+'/stems/'+guitar_id+'.wav'
            if sum([int(i!='') for i in ids]) == n_src:
                break
        
        instru_wav = []

    #     try:
        for p in path:
            wave, sr = librosa.load(p, sr = None)
            instru_wav.append(wave)
    #     except:
    #         error_num += 1
    #         print('error')
    #     #     continue

        min_length = min([len(wav) for wav in instru_wav])
        instru_wav = [wav[:min_length] for wav in instru_wav]
        instru_wav = np.array(instru_wav) # shape - (n_src, min_length)

        length = num_secs * sr
        checkpoint = np.linspace(0, min_length, num=5, endpoint = False)
        for point in checkpoint:
            point = int(point)
            segs = instru_wav[:,point:point+length] # (n_src, seg_length)

            if np.sum(segs,1).all(): # sum of all segments >0
                seg_mixture = np.sum(segs,0)
    #                         print('seg_mixture', seg_mixture.shape)
                sources = segs
                mixtures = seg_mixture
                break

    seg_length = sr = 44100
    overlap = seg_length//10

    window = np.hanning(overlap*2-1) 
    window = np.concatenate((window[:overlap],np.ones(seg_length-overlap*2),window[overlap-1:]))
    window = window.reshape(1,-1)
    window = window.astype(np.float32)

    seg_interval = 44100//10*9
    train_data = []
    for i in range(length//seg_interval):
        seg = mixtures[i*seg_interval:i*seg_interval+seg_length]
        train_data.append(window[0]*seg)
    train_data = torch.tensor(train_data).cuda() # (N, 44100)
#     print(train_data.shape)
    
#     fake()

    # train_data = torch.tensor(train_data).cuda()
    with torch.no_grad():
#         est_sources = []
        est_sources, masked_tf_rep = G_model(train_data) 
        est_remixture = []
        if prop2:
            mask_ratio = torch.tensor(ratio)[None,:,None,None].cuda()
            masked_mixture = torch.unsqueeze(
                    torch.sum(masked_tf_rep * mask_ratio, dim=1), dim=1
                ) #(1, 1, 512, feature_length)
            
            est_remixture = torch.squeeze(G_model.decoder_mix(masked_mixture)) # shape (bt, wav_length)
            est_remixture = pad_x_to_y(est_remixture, est_sources[:,0,:])
#         for seg in train_data:
#             seg = torch.tensor(seg[None, :]).cuda()
#     #         print(seg.shape)
#             est_source, _ = G_model(seg)
#             est_sources.append(est_source.cpu().data.numpy())
#         #     break
#         print(max(torch.flatten(est_sources)))
#         print(torch.max(est_sources[:,1,:]))
    
    iso_sources = []
    
    for i in range(n_src):
        wave_src = np.zeros(length)
#         source = train_data
#         print(est_sources.shape)
        src = est_sources[:,i,:].cpu().data.numpy()# (N, 44100)
        for k in range(length//seg_interval):
            wave_src[k*seg_interval:k*seg_interval+seg_length] += src[k]
        iso_sources.append(wave_src)
    
    iso_sources = np.array(iso_sources)
    
    wave_mix = np.zeros(length)
    mix = est_remixture.cpu().data.numpy()# (N, 44100)
    print(mix.shape)
    for k in range(length//seg_interval):
        wave_mix[k*seg_interval:k*seg_interval+seg_length] += mix[k]
    
#     print(max(iso_sources[0]))
#     print(max(sources[0]))
    
#     pit_loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx') # loss_func(ets, ori)
#     pit_loss_val, reordered_est_sources, batch_indices = pit_loss_func(torch.tensor(iso_sources[None,:,:]), torch.tensor(sources[None,:,:]), return_est=True) 

    return iso_sources, sources, wave_mix, mixtures
#     return reordered_est_sources, sources, mixtures, pit_loss_val
    # est_sources = np.array(est_sources)
    # np.save('../Samples/2sources.npy', est_sources)

if __name__ == '__main__':
    
    iso_sources = eval_wave(n_src=1)
    