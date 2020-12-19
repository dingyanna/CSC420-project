# Load the model
from ImageDataset import MyDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from model import Net1, Net2
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import cv2
import os

def load_data(batch_size):

    test_path = "data/rand_flowers_40/test/"
    test_path_masked = "data/rand_flowers_40_masked/test/"

    test_path = "data/brick_40/test/"
    test_path_masked = "data/brick_40_masked/test/"

    test_dataset=MyDataset(test_path,test_path_masked)
    test_loader=DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

    return test_loader

#net = Net2()
model_name="model_lr_0.001_bs_32_epochs_1600_net_3"
model_path = "models/"+model_name+".pt"
output_folder=os.path.join("tested",model_name)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

net=(torch.load(model_path))

for i,batch in enumerate(load_data(1)):
    images, grounds, fname = batch
    images = images.float()
    output = net(images)
    print("testing")
    cv2.imwrite(os.path.join(output_folder,("tested_"+"brick" + (fname[0]))), output[0].detach().numpy() * 255)

