import musdb
import torch
import numpy as np

MUS_PERM_ID = np.array([ 6, 48, 10, 28, 43, 44,  5, 34,  9, 35, 31, 13, 33, 20,  8, 18, 17,
        1, 38, 29, 16,  0, 24, 15, 47, 19, 36, 14, 12,  3,  7, 26, 39, 41,
       37, 21, 27,  4, 23, 42, 32, 49,  2, 22, 30, 40, 25, 45, 11, 46])
MUS_VALID_ID = MUS_PERM_ID[:25]
MUS_EVAL_ID = MUS_PERM_ID[25:]

DATASET = 'test'
SEC_P_SEG = 1
SR = 44100
mus = musdb.DB(root='/media/sdc1/Data/musdb18', subsets=DATASET)

global_max = 0

all_length = []

for i, track in enumerate(mus):
    
    # take the last 25 for testing
    if i not in MUS_EVAL_ID:
        continue
    print(i)
    sources= []
    
    stems = track.stems #(5, L, 2)
    stems = np.mean(stems, -1) 
    print(i, track.name)
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
    
    seg_length = sr = 44100
    overlap = seg_length//10

    seg_interval = 44100//10*9
    sources = []
    for k in range(length//seg_interval-1):
        seg = instru_wav[:, k*seg_interval:k*seg_interval+seg_length]
#         sources.append(seg) #shape (4, L)
        if np.sum(seg,1).all() and seg.shape[-1]==seg_length: # sum of all segments >0
#             seg_mixture = np.sum(segs,0)
            sources.append(seg)
        
    all_length.append(len(sources))
    
    torch.save(torch.tensor(sources),'../../Data/MUSDB/eval/mus_' + str(i) +'.pt')
#     break
    
config = {'global_max':global_max, 'all_length': all_length}

torch.save(config, '../../Data/MUSDB/eval/config.pt')