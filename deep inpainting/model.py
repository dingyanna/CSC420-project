import torch.nn as nn
import torch.nn.functional as F
import torch

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # in, out, size
        self.pool = nn.MaxPool2d(2, 2)
        kernel_size=5
        self.pad=2
        self.conv1 = nn.Conv2d(3, 10, kernel_size,padding=self.pad)
        self.conv2 = nn.Conv2d(10, 20, kernel_size,padding=self.pad)
        self.conv3 = nn.Conv2d(20, 30, kernel_size,padding=self.pad)
        self.conv4 = nn.Conv2d(30, 20, kernel_size,padding=self.pad)
        self.conv5 = nn.Conv2d(20, 10, kernel_size,padding=self.pad)
        self.conv6 = nn.Conv2d(10, 3, kernel_size,padding=self.pad)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))

        x = F.relu(self.conv6(x))
        x = x.permute(0, 2, 3, 1)

        return x



class Net2(nn.Module):
    # modified from https://debuggercafe.com/autoencoder-neural-network-application-to-image-denoising/
    def __init__(self):
        super(Net2, self).__init__()
        # in, out, size
        # encoder layers


        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)
        # was 3 here
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # encode
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x)

        # decode
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.sigmoid(self.out(x))
        x = x.permute(0, 2, 3, 1)

        return x

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        # in, out, size
        # encoder layers


        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)
        # was 3 here
        self.dec2 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # encode
        x = x.permute(0, 3, 1, 2)
        x1=self.enc1(x) # BS,64,40,40
        x = F.relu(x1)
        x = self.pool(x)
        x2=self.enc2(x) # BS,32,20,20
        x = F.relu(x2)
        x = self.pool(x)
        x3=self.enc3(x) #BS, 16, 10, 10
        x = F.relu(x3)
        x = self.pool(x)
        x4=self.enc4(x) # BS, 8, 5,5
        x = F.relu(x4)
        x = self.pool(x)

        # decode

        x = F.relu(torch.cat((self.dec1(x), x4), dim=1))
        x = F.relu(torch.cat((self.dec2(x), x3), dim=1))
        x = F.relu(torch.cat((self.dec3(x), x2), dim=1))
        x = F.relu(torch.cat((self.dec4(x), x1), dim=1))
        x = F.sigmoid(self.out(x))
        x = x.permute(0, 2, 3, 1)

        return x