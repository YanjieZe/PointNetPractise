from pcdataloader import pcdataset,DataLoader
from partnet import partnet
from semnet import semnet
import torch
import torch.optim

learning_rate = 0.001
batch_size = 5
part_num = 1
validation_split = 0.2
shuffle_dataset = True


model = semnet()

optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            eps=1e-08,
            weight_decay=True
        )

dataset = pcdataset(transform=None)

if torch.cuda.is_available():
    devide = torch.device("cuda")
    

"""
part segmentation任务的输出是什么？
到底什么是part segmentation？

还剩下未做的：
数据预处理
数据集的label没有设置
搞清楚什么part segmentation
进行训练
"""


loader = DataLoader(dataset,batch_size)