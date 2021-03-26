import os
import torch
from torch.utils import data
import librosa
import numpy as np
from IPython.display import clear_output
import glob

class Slakh_data(data.Dataset):
    
    def __init__(self, dataset, n_src, length_p_sec, piano, drum, bass, guitar, size):
        
        folderPath = '/media/sdc1/slakh2100_wav/'+dataset+'/*' 
        
        self.mixtures = []
        self.sources = []
        self.global_max = 0
        
        error_num = 0
        
        for idx, track in enumerate(glob.glob(folderPath)):
            print(track)
        
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
                    error_num += 1
                    continue
                    
                min_length = min([len(wav) for wav in instru_wav])
                instru_wav = [wav[:min_length] for wav in instru_wav]
                instru_wav = np.array(instru_wav) # shape - (n_src, min_length)
                std_wav = np.std(np.sum(instru_wav, 0))
                instru_wav /= std_wav
                max_wav = np.max(np.sum(instru_wav, 0))
                # Using standard deviation and max value from the entire wav
            
                if self.global_max < max_wav:
                    self.global_max = max_wav
                
#                 fake()
                # normalized for the standard deviation of the entire wave
                
                seg_length = length_p_sec * sr
                checkpoint = np.linspace(0, min_length, num=15, endpoint = False)
                for point in checkpoint:
                    point = int(point)
#                     seg0 = instru_wav[0][point:point+seg_length]
# #                     print('seg0', seg0.shape)
#                     seg1 = instru_wav[1][point:point+seg_length]
                    segs = instru_wav[:,point:point+seg_length] # (n_src, seg_length)
    
                    if np.sum(segs,1).all(): # sum of all segments >0
                        seg_mixture = np.sum(segs,0)
#                         print('seg_mixture', seg_mixture.shape)
                        self.sources.append(segs)
                        self.mixtures.append(seg_mixture)
            
            if idx - error_num >size:
                self.sources /= self.global_max
                self.mixtures /= self.global_max
                print(self.global_max)
                break
                
            
    def __len__(self):
        return len(self.mixtures)

    def __getitem__(self, idx):
        return self.mixtures[idx], self.sources[idx]
                
                        
                        
                
                

