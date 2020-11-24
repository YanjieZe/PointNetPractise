from pcdataloader import pcdataset,DataLoader,pc_normalize
from partnet import partnet
from semnet import semnet
import torch
import torch.optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os

learning_rate = 1e-8
batch_size = 3
part_num = 1
validation_split = 0.2
shuffle_dataset = True
epoch = 50
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = semnet().float().to(device)

if os.path.exists('params.pkl'):
    model.load_state_dict(torch.load('params.pkl'))

optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            eps=1e-08,
        )

dataset = pcdataset(transform=pc_normalize)

loss_fn = nn.MSELoss()
loader = DataLoader(dataset,batch_size)




model.train()
for e in range(epoch):
    for idx,(xtrain,ytrain) in enumerate(loader):
        
        x_train = xtrain.float().to(device)
        y_train = ytrain.float().to(device)

        result,_ = model(x_train)
        loss = loss_fn(result,y_train)
        print("epoch: ",e," idx: ",idx, " loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'params.pkl')
