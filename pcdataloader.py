import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import warnings
warnings.filterwarnings("ignore")


def loadnpy1():
    data = np.load("./hw/0.npy")
    data = data.T
    data = np.expand_dims(data, axis=0)
    judge = 1
    for i in range(19):

        if judge ==1:
            i=i+1
            judge+=1

        filename  ="./hw/%d.npy"%i
        a = np.load(filename)
        a = a.T
        a= np.expand_dims(a,axis=0)
        data = np.vstack([data,a])
    
    return data

def loadnpy2():
    data = np.load("./hw/0.npy")
  
    data = np.expand_dims(data, axis=0)
    judge = 1
    for i in range(19):

        if judge ==1:
            i=i+1
            judge+=1

        filename  ="./hw/%d.npy"%i
        a = np.load(filename)
       
        a= np.expand_dims(a,axis=0)
        data = np.vstack([data,a])
    
    return data

# point cloud dataset
class pcdataset(Dataset):

    def __init__(self, transform=None,split='train'):
        self.data = loadnpy1()
        self.transform = transform
        self.label = loadnpy2()

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_i = self.data[idx]
        label_i = self.label[idx]
        
        if self.transform:
            data_i = self.transform(data_i)
        return data_i,label_i

    def __len__(self):
        return self.data.shape[0]

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


if __name__ =="__main__":
    
    set = pcdataset()
    loader = DataLoader(set, batch_size=4, shuffle=True)
    for i,(x,y) in enumerate(loader):
        print(x.shape,y.shape)
        break



