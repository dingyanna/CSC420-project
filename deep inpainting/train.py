from ImageDataset import MyDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from model import Net1, Net2, Net3
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def load_data(batch_size):

    train_path = "data/rand_flowers_40/train/"
    train_path_masked = "data/rand_flowers_40_masked/train/"
    val_path = "data/rand_flowers_40/val/"
    val_path_masked = "data/rand_flowers_40_masked/val/"

    train_dataset=MyDataset(train_path,train_path_masked)
    val_dataset=MyDataset(val_path,val_path_masked)

    train_loader=DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset, batch_size=batch_size,shuffle=False)

    return train_loader, val_loader

def loss_pixcel_diff(y_true, y_pred):
   return torch.abs(y_true - y_pred)

def load_model(lr):
    model = Net3()
    loss_fnc = nn.MSELoss()
    #loss_fnc = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer

def evaluate(model, data_loader, loss_fnc=nn.MSELoss()):
    total_loss = 0
    for i,batch in enumerate(data_loader):
        images, grounds, _ = batch
        images = images.float()
        predictions = model(images)
        batch_loss = loss_fnc(predictions, grounds.float())
        total_loss += batch_loss

    return total_loss / len(data_loader)


def main():

    torch.manual_seed(1000)
    batch_size=32
    lr=0.0001
    epochs=1600
    eval_every=20

    train_loader, val_loader= load_data(batch_size)
    model, loss_fnc, optimizer = load_model(lr)

    N=0
    val_loss_list = []
    epoch_list = []
    train_loss_list=[]


    for epoch in range(epochs):

        accum_loss = 0.0

        for i, batch in enumerate(train_loader):

            images, grounds, _ =batch
            optimizer.zero_grad()
            images = images.float()
            predictions = model(images)
            batch_loss = loss_fnc(predictions, grounds.float())
            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

            #print("batch ",i)

        N += 1

        if (N % (eval_every) == 0):

            epoch_list.append(epoch)
            if len(val_loss_list) > 0:
                best_last_time = min(val_loss_list)
            val_loss = evaluate(model, val_loader)
            if len(val_loss_list) > 0 and val_loss < best_last_time:
                print("saving at",epoch)
                torch.save(model, "./models/rand_model_lr_{}_bs_{}_epochs_{}_net_{}.pt".format(lr,batch_size,epochs,"3"))

            val_loss_list.append(val_loss.item())
            train_loss_list.append(accum_loss.item()/len(train_loader))

            df = pd.DataFrame({"epoch": epoch_list, "val_loss": val_loss_list, "train_loss": train_loss_list})
            df.to_csv("./models/rand_loss_lr_{}_bs_{}_epochs_{}_net_{}.csv".format(lr, batch_size, epochs, "3"))

            print("Epoch {}: train loss: {}, validation loss: {}".format(epoch,accum_loss/len(train_loader), val_loss))
            df = pd.DataFrame( {"epoch": epoch_list, "val_loss": val_loss_list, "train_loss": train_loss_list})
            #df.to_csv("loss_for_plot3.csv")


if __name__ == "__main__":

    main()
