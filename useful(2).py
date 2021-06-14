import torch
from torchvision import transforms, models
import argparse
import json
from torch import nn

from assist import process_image
import argparse


def load_checkpoint(arch, hidden, filepath):
    checkpoint = torch.load(filepath)
    
    if arch =='resnet50':
        model = models.resnet50(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
      
        classifier = nn.Sequential(nn.Linear(2048,hidden),
                                nn.ReLU(),

                                nn.Linear(hidden, 102),

                                nn.LogSoftmax(dim=1))

        model.fc = classifier
        
    elif arch == 'VGG':
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
        
        classifier = nn.Sequential(nn.Linear(25088,4096),
                                nn.ReLU(),
                                   
                                   nn.Linear(4096, hidden),
                                   nn.ReLU(),

                                nn.Linear(hidden, 102),

                                nn.LogSoftmax(dim=1))

        
        
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint ['class_to_idx']
    
    return model


def predict(image_path, model, topk):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    image = process_image(image_path)
    x = image.numpy()

    image = torch.from_numpy(x).float()
    
    image = image.unsqueeze(0)
    image = image.cuda()
    model.to(device)

    output = model.forward(image)
    
    ps = torch.exp(output).data
 
    largest = ps.topk(topk)
    prob = largest[0].cpu().numpy()[0]
    idx = largest[1].cpu().numpy()[0]
    classes = []
    idx = largest[1].cpu().numpy()[0]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    for i in idx:
        classes.append(idx_to_class[i])
    
    return prob, classes

#def predict(image_path, model, topk):
#    
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
#    model = model.to(device)
    
#    processed_image = process_image(image_path)
#    processed_image = processed_image.type(torch.FloatTensor)
    
#    with torch.no_grad():
    
#        model.eval()
        
#        processed_image = processed_image.unsqueeze_(0)
        
#        processed_image = processed_image.to(device)
        
#        output = model.forward(processed_image)
#        ps = torch.exp(output).data
        #ps = torch.exp(model(processed_image.to(device)))
    
#        top_p, top_class = ps.topk(topk, dim=1)
        
#        probs = top_p.cpu().numpy()
#        indexs = top_class.cpu().numpy()[0]
#        classes = []
        
#        for i in indexs:
#            classes.append(idx2class[i])
    
#        return probs, classes