import torch.utils.data as data
import numpy as np
import cv2
import glob



class MyDataset(data.Dataset):

    def __init__(self, original_path, mask_path):
        original_list = glob.glob(original_path+'/*.jpg')
        mask_list = glob.glob(mask_path + '/*.jpg')

        self.X = np.array([cv2.imread(fname) for fname in mask_list])
        self.y = np.array([cv2.imread(fname) for fname in original_list])
        self.filenames=[fname.split("/")[-1] for fname in original_list]
        # normalize
        self.X=self.X/255
        self.y = self.y / 255

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        masked_image=self.X[index]
        ground=self.y[index]
        f=self.filenames[index]
        return masked_image,ground,f
