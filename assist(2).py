import numpy as np
import matplotlib.pyplot as plt
import os, random
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time

def read(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(25),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    val_test_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=val_test_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=val_test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    
    return trainloader, testloader, valid_loader, train_data

def process_image(image):
    
    image = Image.open(image)
    image = image.resize((256,256))
    image = image.crop((0,0,224,224))
    np_image = np.array(image)/255
    mean = np.array([0.485,0.456,0.406])
    stdv = np.array([0.229, 0.224, 0.225])
    image = (np_image - mean) / stdv
    p_image = image.transpose((2,1,0))
    
    return torch.from_numpy(p_image)
    
  