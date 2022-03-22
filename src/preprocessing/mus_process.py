import musdb
import torch
import numpy as np

DATASET = 'test'
SEC_P_SEG = 1
SR = 44100
# mus = musdb.DB(root='/media/sdc1/Data/musdb18', subsets=DATASET)
mus = musdb.DB(root='/media/sdb1/Data/musdb18_32khz', subsets=DATASET)

global_max = 0

all_length = []

print(len(mus))

for i, track in enumerate(mus):

    sources= []
    
    stems = track.stems #(5, L, 2)
    stems = np.mean(stems, -1) 
    print(i, track.name)
    fake()
    # (5, L) mixture, drums, bass, the rest, vocal
    length = stems.shape[-1]
    instru_wav = stems[[-1, 1, 2, 3]]
    std_wav = np.std(np.sum(instru_wav, 0))
    instru_wav /= std_wav
    max_wav = np.max(np.sum(instru_wav, 0))
    # Using standard deviation and max value from the entire wav
    
    if global_max < max_wav:
        global_max = max_wav
    # normalized for the standard deviation of the entire wave
        
    seg_length = SEC_P_SEG * SR
#             checkpoint = np.linspace(0, length, num=30, endpoint = False)
    for point in range(0, length, seg_length):
        point = int(point)
        segs = instru_wav[:,point:point+seg_length] # (n_src, seg_length)
#                 print(segs.shape)
        if np.sum(segs,1).all() and segs.shape[-1]==seg_length: # sum of all segments >0
            seg_mixture = np.sum(segs,0)
            sources.append(segs)
#     print(len(sources))
    all_length.append(len(sources))
#     print(sources[0].shape)

    torch.save(torch.tensor(sources),'../../Data/MUSDB/'+DATASET+'/mus_' + DATASET + '_' + str(i) +'.pt')
#     break
    
# config = {'global_max':global_max, 'all_length': all_length}

# torch.save(config, '../../Data/MUSDB/'+DATASET+'/config.pt')