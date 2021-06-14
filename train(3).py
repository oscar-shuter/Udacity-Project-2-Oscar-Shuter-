import torch
import time
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import argparse
import json
from collections import OrderedDict

from assist import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action='store', help='directory containing images', default = 'flowers')
    parser.add_argument('--save_dir', action='store', help='save trained checkpoint to this directory', default = 'checkpoint.pth' )
    parser.add_argument('--arch', action='store', help='which neural network: resnet50 or VGG', default='resnet50')
    parser.add_argument('--gpu', action='store_true', help='use gpu to train model')
    parser.add_argument('--epochs', action='store', help='No. of epochs for training', type=int, default=4)
    parser.add_argument('--lr', action='store', help='initial learning rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', help='No. of hidden units in model', type=int, default=500)
    parser.add_argument('--output_size', action='store', help='No. of output units in model', type=int, default=102)
    
    return parser.parse_args()



def main():
  
    in_arg = get_input_args()
    
    start_time = time.time()
    
    dataloaders = load_data(in_arg.data_dir)
    
    model = get_model(in_arg.arch)
        
    model = load_model(in_arg.arch, in_arg.hidden_units, in_arg.lr)

    criterion = nn.NLLLoss()
    
    if in_arg.arch == 'resnet50':
        optimizer = optim.Adam(model.fc.parameters(), lr=in_arg.lr)
    elif in_arg.arch == 'VGG':
        optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.lr)
    
    
    
    
    train(model, in_arg.epochs, criterion, optimizer, dataloaders[0], dataloaders[2],in_arg.gpu)
   

    #save_checkpoint(in_arg.save_dir, model, optimizer, in_arg.epochs, in_arg.arch, dataloaders, in_arg.lr)

    model.class_to_idx = dataloaders[-1].class_to_idx
    checkpoint = {'output_size' : 102,
                  'optimizer': optimizer,
                  'epochs': in_arg.epochs,
                  'state_dict': model.state_dict(),
                  'optimizer_state': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'classifier': classifier.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')
    
def get_model(arch):
    if arch == 'resnet50': #
        model = models.resnet50(pretrained = True)
        
    elif arch == 'VGG':
        model = models.vgg16(pretrained = True)
    return model



def load_model(arch, hidden_units, lr):
    
    global classifier

    if arch == 'resnet50':    
        model=models.resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        classifier= nn.Sequential(nn.Linear(2048,hidden_units),
                              nn.ReLU(),
                              nn.Linear(hidden_units, 102),
                              nn.LogSoftmax(dim=1))
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    

    elif arch == 'VGG': 
            model = models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(4096, hidden_units)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.5)),
            ('fc3', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))
            model.classifier = classifier
            optimizer = optim.Adam(model.classifier.parameters(), lr=lr)


    criterion = nn.NLLLoss()
    
    model.to(device)
    return model

def train(model, epochs, criterion, optimizer, trainloader, valid_loader, gpu):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model.train() 
    epochs=epochs 
    steps=0
    running_loss=0
    print_every=10
    print(device)
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps+=1
            inputs, labels =inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            
            if steps % print_every==0:
                test_loss=0
                accuracy=0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in  valid_loader:
                        inputs, labels=inputs.to(device), labels.to(device)
                        logps =model.forward(inputs)
                        batch_loss=criterion(logps, labels)
                        test_loss+=batch_loss.item()
                        ps=torch.exp(logps)
                        top_p, top_class=ps.topk(1, dim=1)
                        equals = top_class==labels.view(*top_class.shape)
                        accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.."
                      f"Test loss:{running_loss/print_every:.3f}.."
                      f"Validation loss:{test_loss/len(valid_loader):.3f}.."
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")  
                
                running_loss = 0
                model.train()

def validate(model, criterion, valid_loader):
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for image, label in valid_loader: 
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
        return ('Test Accuracy: %d %%' % (100 * correct / total))
    

#def save_checkpoint(save_dir, model, optimizer, epochs, arch, image_datasets, lr):
#    model.class_to_idx = image_datasets[-1].class_to_idx
#    checkpoint = {'output_size' : 102,
#                  'optimizer': optimizer,
#                  'epochs': epochs,
#                  'arch': arch,
#                  'state_dict': model.state_dict(),
#                  'optimizer_state': optimizer.state_dict(),
#                  'class_to_idx': model.class_to_idx,
#                  'classifier': classifier.state_dict()}
#    torch.save(checkpoint, save_dir)
    
       
   


if __name__ == "__main__":
    main()











