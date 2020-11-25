import torch
from semnet import semnet
from pcdataloader import loadnpy2,loadnpy1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = semnet().float().to(device)

model.load_state_dict(torch.load('params.pkl'))

data = torch.from_numpy(loadnpy1()).float().to(device) # 20*3*2048
pointcloud = torch.from_numpy(loadnpy2()).float().to(device) # 20*2048*3
result = model(data)

print(result - pointcloud)