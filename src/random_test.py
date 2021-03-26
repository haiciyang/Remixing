import asteroid
import torch
from asteroid.models import ConvTasNet
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# pit_loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx') # loss_func(ets, ori)
# x = torch.rand(5, 3, 200)
# y = torch.stack((x[:,0,:], x[:,2,:], x[:,1,:]), dim=1)

# pit_loss_val, reordered_est_sources, batch_indices = pit_loss_func(y, x, return_est=True) 
# print(batch_indices)

G_model = ConvTasNet(n_src=2).cuda()
print(model.enc)