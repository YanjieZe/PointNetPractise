import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet import PointNetEncoder, feature_transform_reguliarzer
import numpy as np


class semnet(nn.Module):
    def __init__(self,num_class=3,with_rgb=False):
        super(semnet,self).__init__()
        if with_rgb:
            channel = 6
        else:
            channel = 3
        self.k = num_class
        self.feat = PointNetEncoder()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=channel)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
    
    def forward(self,x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()#
        # print("2:",x.shape)
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # print("3:",x.shape)
        # x = x.view(batchsize, n_pts, self.k)
        # print("4:",x.shape)
        return x

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight = weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == "__main__":
    model = semnet(3,False).float()
    xyz = torch.rand(2,3,2048)
    
    print(xyz.shape)
    result= model(xyz.float())
    print(result.shape)