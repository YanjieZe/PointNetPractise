import torch
from semnet import semnet
from pcdataloader import loadnpy2,loadnpy1
from visualize import visualize


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = semnet().float().to(device)

model.load_state_dict(torch.load('params.pkl'))

with torch.no_grad():
    data = torch.from_numpy(loadnpy1()).float().to(device) # 20*3*2048
    pointcloud = torch.from_numpy(loadnpy2()).float().to(device) # 20*2048*3
    result = model(data)

pc_trained = result.to("cpu").numpy()
pc_origin = loadnpy2()
visualize(pc_trained[0])