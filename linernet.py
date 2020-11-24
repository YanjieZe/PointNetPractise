import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet import STN3d

if __name__ == "__main__":
    arr = torch.rand(2,2048,3)
    net = STN3d(3)
    res = net(arr)
    print(res)# batch*3*3