import torch
from torchvision import transforms, models
import argparse
import json

from assist import process_image
import argparse
from useful import load_checkpoint
from useful import predict
from assist import process_image

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action='store', help='path to image to be classified')
    parser.add_argument('checkpoint', action='store', help='path to stored model')
    parser.add_argument('--top_k', action='store', type=int, default=1, help='how many most probable classes to print out')
    parser.add_argument('--category_names', action='store', help='file which maps classes to names', default = 'cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', help='use gpu to infer classes')
    parser.add_argument('--arch', action='store', help='which neural network: resnet50 or VGG', default='resnet50')
    parser.add_argument('--hidden_units', action='store', help='No. of hidden units', default=500)
    args=parser.parse_args()
    
    return parser.parse_args()
    
def main():

    in_arg = get_input_args()    
    
    model = load_checkpoint(in_arg.arch, in_arg.hidden_units, in_arg.checkpoint)
    
    processed_image = process_image(in_arg.input) 
    
    top_probs, top_labels = predict(in_arg.input, model, in_arg.top_k)
    
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    top_name = []
    for i in range(len(top_labels)):
        
        top_name.append(cat_to_name[top_labels[i]])
        
    print('Most likely class/classes: ' , top_labels, top_name)
    print('Category likelihood/s: ' , top_probs)
    
    
    
        
    
 
if __name__ == "__main__":
    main()
            
    
    