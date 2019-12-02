import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn


class MultiLabelClassifier(torch.nn.Module):
    def __init__(self, FeatureExtractor):
        """
        A simple 1d vector encoder
        Inputs:
            D_in: Size of the input vector
            D_out: Size of the embedding space
        """
        super(MultiLabelClassifier,self).__init__()
        self.FeatureExtractor =  nn.Sequential(*list(torch.hub.load('pytorch/vision:v0.4.2', FeatureExtractor, pretrained=True,  progress=True).children())[:-2])
        
        self.conv1 = torch.nn.Conv2d(512, 64, 7, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.linear = torch.nn.Linear(64, 6, bias=True)
        
        
    def forward(self,x):
        """
        Inputs: a 2d tensor of size [H,W]
        Returns: a 1d tensor of dimension [10]
        """
        x = self.FeatureExtractor(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.sigmoid(self.linear(x.squeeze()))
        return x
