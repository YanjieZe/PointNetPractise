import torch

a = torch.rand(2,5)
b = torch.rand(2,2).squeeze(1)
res = torch.cat([a,b],1)
print(b.shape)
print(res.shape)