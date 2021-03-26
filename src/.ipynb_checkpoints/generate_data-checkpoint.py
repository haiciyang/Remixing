import argparse
from dataset import Slakh_data
import torch

parser = argparse.ArgumentParser(description='Generating data')


parser.add_argument('--n_src', type=int, default=2,  help='Number of sources');
parser.add_argument('--length_p_sec', type=int, default=1,  help='Input length to the network');
parser.add_argument('--batch', type=int, default=2,  help='Input length to the network');
parser.add_argument('--dataset', type=str, default='test', help='dataset to process')
parser.add_argument('--size', type=int, default=100, help='dataset to process')
parser.add_argument('--save_name', type=str, default='',  help='Save the output data');

parser.add_argument('--piano', dest='piano', action='store_true', help='extract piano')
parser.add_argument('--drum', dest='drum', action='store_true', help='extract drum')
parser.add_argument('--bass', dest='bass', action='store_true', help='extract bass')
parser.add_argument('--guitar', dest='guitar', action='store_true', help='extract guitar')


args = parser.parse_args()


data = Slakh_data(args.dataset, args.n_src, args.length_p_sec, args.piano, args.drum, args.bass, args.guitar, args.size)
print(len(data))

loader = torch.utils.data.DataLoader(data, batch_size = args.batch, shuffle = True, num_workers = 4)

torch.save(loader, '../Data/'+args.save_name+'.pth')
