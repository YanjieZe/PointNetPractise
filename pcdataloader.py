import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import warnings
warnings.filterwarnings("ignore")


def loadnpy():
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
        self.data = loadnpy()
        self.transform = transform

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_i = self.data[idx]
        
        if self.transform:
            data_i = self.transform(data_i)
        return data_i

    def __len__(self):
        return self.data.shape[0]




if __name__ =="__main__":
    
    set = pcdataset()
    loader = DataLoader(set, batch_size=4, shuffle=True)
    for i,da in enumerate(loader):
        print(da.shape)
        break



