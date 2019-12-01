from torchvision import transforms
from torch.utils import data
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from DataLoader import *

TrainLoader = TrafficLight(RootDirectory="./data/train/")
Params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
TrainingGenerator = data.DataLoader(TrainLoader, **Params)

TestLoader = TrafficLight(RootDirectory="./data/test/")
Params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
TestingGenerator = data.DataLoader(TestLoader, **Params)

Model = MultiLabelClassifier(FeatureExtractor = 'resnet18')
Loss = nn.BCELoss(reduction='mean')
Optimizer = torch.optim.Adam(params = Model.parameters())

for epoch in range(10):
    # Training
    t3 = time.time()
    for local_batch, local_labels in TrainingGenerator:
        t1 = time.time()
        Model.zero_grad()
        
        Output = Model(local_batch)
        t2 = time.time()
        
        l = Loss(Output, local_labels)
        l.backward()
        Optimizer.step()
        
        t3 = time.time()
        print("Batch Load Time :", t3-t1,"Forward Pass Time:", t2 - t1, "Optimization time:", t3-t2, "Loss", l.item() )