from torchvision import transforms
from torch.utils import data
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from Models import *
import cv2 as cv2
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help="image whose traffic ligt state is to be known")
parser.add_argument('-c', '--checkpoint', help="Checkpoint for the model")
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
Model = MultiLabelClassifier(FeatureExtractor = 'resnet50').to(device)
Model.load_state_dict(torch.load(args.checkpoint))
Model.eval()

States = ["red", "yellow", "green", "green-left", "green-right", "green-solid"]

Model.zero_grad()
Transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

Img = torch.unsqueeze(Transform(Image.open(args.image)),0)

Output = Model(Img)

for idx, state in enumerate(States):
    print("Probability of state {} being actiive is ".format(state), float(Output.detach()[0,idx]))