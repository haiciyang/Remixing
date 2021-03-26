import os
import torch
from torch.utils import data
import librosa
import numpy as np
from IPython.display import clear_output
import glob

class Data(data.Dataset):
    
    def __init__(self, ):
        
        folderPath = '/media/sdc1/slakh2100_wav/train'
        
        for track in glob.glob('/media/sdc1/slakh2100_wav/train/*'):
            
        
        import glob

        piano_path = ''
        bass_path = ''

        for track in glob.glob('/media/sdc1/slakh2100_wav/train/*'):

            with open(track+'/metadata.yaml') as f:
                lines = f.readlines()
                piano_id = ''
                bass_id = ''

                for i, line in enumerate(lines):
                    if 'Piano' in line and not piano_id:
                        piano_id = lines[i-2][-5:-2]
                    if 'Drum' in line and not bass_id:
                        bass_id = lines[i-2][-5:-2]
                    if piano_id and bass_id:
                        break

                piano_path = track+'/stems/'+piano_id+'.wav'
                bass_path = track+'/stems/'+bass_id+'.wav'

                print(piano_wav)
            break