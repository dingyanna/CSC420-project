# Load the model
from ImageDataset import MyDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from model import Net1
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import cv2

def load_data(batch_size):

    val_path = "data/dummy_flowers_400/test/"
    val_path_masked = "data/dummy_flowers_400_masked/test/"


    val_dataset=MyDataset(val_path,val_path_masked)

    val_loader=DataLoader(val_dataset, batch_size=batch_size,shuffle=False)

    return val_loader

net = Net1()
model_path = "models/model_lr_0.001_bs_32_epochs_300_function_BCE.pt"
net=(torch.load(model_path))
for i,batch in enumerate(load_data(1)):
    images, grounds = batch
    images = images.float()
    output = net(images)



    cv2.imwrite("tested"+str(i)+".jpg",output[0].detach().numpy()*255)