import glob
import torch
import librosa
import numpy as np
import os


# path = '../Data/Slakh/'+DATASET+'/'

# config = torch.load(path + 'config.pt')
# all_length = config['all_length']

# for i, l in enumerate(all_length):
#     if l == 0 :
#         print('delete file',i, l)
#         os.remove(path + str(i) + '.pt') 

DATASET = 'test'
SEC_P_SEG = 1
SR = 44100

folderPath = '/media/sdc1/slakh2100_wav/'+DATASET+'/*' 
global_max = 0
all_length = []
total_file = 100
n_src = 5

for i, track in enumerate(glob.glob(folderPath)):
    
    if i < 100:
        continue
    
    print('Loading', i)

    piano = True 
    drum = True if n_src >=2 else False
    bass = True if n_src >=3 else False
    guitar = True if n_src >=4 else False
    strings = True if n_src >=5 else False

    piano_id = ''
    drum_id = ''
    bass_id = ''
    guitar_id = ''
    strings_id = ''
    ids = [piano_id, drum_id , bass_id , guitar_id, strings_id] 
    path = [''] * n_src
    print(track)
    sources = []
    
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
            if strings and 'Strings' in line and not strings_id:
                strings_id = lines[i-2][-5:-2]
                path[4] = track+'/stems/'+strings_id+'.wav'
            if sum([int(i!='') for i in ids]) == n_src:
                break

#                 piano_path = track+'/stems/'+piano_id+'.wav' if piano_id else ''
#                 drum_path = track+'/stems/'+drum_id+'.wav' if drum_id else ''
#                 bass_path = track+'/stems/'+bass_id+'.wav' if bass_id else ''
#                 guitar_path = track+'/stems/'+guitar_id+'.wav' if guitar_id else ''

#                 paths = [piano_path, drum_path , bass_path , guitar_path] 
        
    instru_wav = []
    
    try:
        for p in path:
    #                         print(p)
            wave, sr = librosa.load(p, sr = None)
            instru_wav.append(wave)
            
    except:
        print('error1')
#         error_num += 1
        continue
#     fake()
    
        
    length = min([len(wav) for wav in instru_wav])
    instru_wav = [wav[:length] for wav in instru_wav]
    try:
        instru_wav = np.array(instru_wav) # shape - (n_src, min_length)
    except:
        print('error2')
#         error_num += 1
        continue
#                 print(instru_wav.shape)
    std_wav = np.std(np.sum(instru_wav, 0))
    instru_wav /= std_wav
    max_wav = np.max(np.sum(instru_wav, 0))
    # Using standard deviation and max value from the entire wav

    if global_max < max_wav:
        global_max = max_wav
    
#                 fake()
    # normalized for the standard deviation of the entire wave
    
    seg_length = SEC_P_SEG * SR
    # checkpoint = np.linspace(0, min_length, num=50, endpoint = False)
    overlap = seg_length//10

    seg_interval = 44100//10*9
    sources = []
    for i in range(length//seg_interval-1):
        seg = instru_wav[:, i*seg_interval:i*seg_interval+seg_length]
        sources.append(seg) #shape (4, L)
    
#     if not sources:
    print(len(sources))
    all_length.append(len(sources))
    torch.save(torch.tensor(sources),'../../Data/Slakh/eval_5/'+str(total_file) +'.pt')
    total_file += 1
#     else:
#         print('Empty array')
    
#     break

config = {'global_max':global_max, 'all_length': all_length}
torch.save(config,'../../Data/Slakh/eval_5/config.pt')