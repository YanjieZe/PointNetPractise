from pcdataloader import pcdataset,DataLoader
from partnet import partnet
import torch

learning_rate = 0.001
batch_size =5
part_num = 1
validation_split = 0.2


model = partnet(part_num)

optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            b3etas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=True
        )

dataset = pcdataset(transform=None)
loader = DataLoader(dataset,batch_size)