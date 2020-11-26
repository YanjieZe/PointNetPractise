from pcdataloader import pcdataset,DataLoader,pc_normalize,loadnpy1,loadnpy2
from partnet import partnet
from semnet import semnet
import torch
import torch.optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os

learning_rate = 1e-6
batch_size = 2
shuffle_dataset = True
epoch = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = semnet().float().to(device)


model.load_state_dict(torch.load('params.pkl'))

optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            eps=1e-08,
        )

dataset = pcdataset(transform=pc_normalize)

loss_fn = nn.MSELoss()
loader = DataLoader(dataset,batch_size)
train_log = open('train_log.txt','w')

model.train()
for e in range(epoch):
    for idx,(xtrain,ytrain) in enumerate(loader):
        
        x_train = xtrain.float().to(device)
        y_train = ytrain.float().to(device)

        result = model(x_train)
        loss = loss_fn(result,y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_log.write("epoch: "+str(e)+" loss:"+str(loss.item())+'\n')
    print("epoch: ",e," loss:", loss.item())

torch.cuda.empty_cache()
print("---------------------")

# calculate the finnal loss
data = torch.from_numpy(loadnpy1()).float().to(device) # 20*3*2048
pointcloud = torch.from_numpy(loadnpy2()).float().to(device) # 20*2048*3
result = model(data)
final_loss = loss_fn(result,pointcloud)
print("final total loss is: ", final_loss.item())

train_log.close()
torch.save(model.state_dict(), 'params.pkl')
print("---------------------")
print("Save model params finished.")
