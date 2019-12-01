import torch
import glob
import numpy as np
import json
import cv2 as cv2
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils import data
import torch
from Models import *
import time

class TrafficLight():
    """Face Landmarks dataset."""

    def __init__(self, RootDirectory):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ImageList = glob.glob(RootDirectory + '*.jpg')
        self.AvailableLabelList = glob.glob(RootDirectory + '*.json')
        self.NumElems = len(self.ImageList)
        self.Images = np.zeros((self.NumElems, 3, 200, 200))
        self.Labels = np.zeros((self.NumElems, 6))
        self.Transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        i=0
        for ImageName in tqdm(self.ImageList, total=self.NumElems):
            if ImageName.replace(".jpg", ".json") in self.AvailableLabelList:
                self.Images[i,:,:,:] = cv2.imread(ImageName).transpose(2, 0, 1)
                with open(ImageName.replace(".jpg", ".json")) as f:
                    d = json.load(f)
                    self.Labels[i,:] = np.array(d["class"])
                    i+=1
                

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, idx):
        UIntArray = self.Images[idx].astype('uint8')
        PillowImage = Image.fromarray(UIntArray, 'RGB')
        return self.Transform(PillowImage), torch.tensor(self.Labels[idx], dtype=torch.float32)


mod = MultiLabelClassifier(FeatureExtractor = 'resnet18')
