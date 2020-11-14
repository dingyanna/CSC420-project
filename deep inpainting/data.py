import glob
import cv2
import numpy as np
import os

# data source https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html

def mask(image,size,seed=36):
    np.random.seed(seed)
    mask = np.full((size, size, 3), 255, np.uint8)
    size = size
    scratches = 1
    thickness = 2
    for i in range(scratches):
        x1, x2, y1, y2 = np.random.randint(1, size), np.random.randint(1, size), np.random.randint(1, size), np.random.randint(1, size)
    cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 0), thickness)
    masked_image = image.copy()
    masked_image[mask == 0] = 255
    return masked_image, mask

def train_val_test_split(L, train_percent=0.6, validate_percent=0.2, seed=36):
    np.random.seed(seed)
    np.random.shuffle(L)
    train_end = int(train_percent * len(L))
    validate_end = int(validate_percent * len(L)) + train_end
    train = L[:train_end]
    validate = L[train_end:validate_end]
    test = L[validate_end:]
    return train, validate, test

resize=40
filelist = glob.glob('data/jpg/*.jpg')
if resize==None:
    x = np.array([np.array(cv2.imread(fname)[0:400,0:400,:]) for fname in filelist])
else:
    x = np.array([np.array(cv2.resize(cv2.imread(fname)[0:400, 0:400, :],(resize,resize)) )for fname in filelist])

train, val, test=train_val_test_split(x)

train_path="data/flowers_{size}/train/".format(size=resize)
train_path_masked="data/flowers_{size}_masked/train/".format(size=resize)
val_path="data/flowers_{size}/val/".format(size=resize)
val_path_masked="data/flowers_{size}_masked/val/".format(size=resize)
test_path="data/flowers_{size}/test/".format(size=resize)
test_path_masked="data/flowers_{size}_masked/test/".format(size=resize)

dirs=[train_path,train_path_masked,val_path,val_path_masked,test_path,test_path_masked]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

for i in range(len(train)):
    cv2.imwrite(train_path+filelist[i].split("/")[-1],x[i])
    cv2.imwrite(train_path_masked + filelist[i].split("/")[-1], mask(x[i],resize)[0])
for i in range(len(val)):
    cv2.imwrite(val_path+filelist[i].split("/")[-1],x[i])
    cv2.imwrite(val_path_masked + filelist[i].split("/")[-1], mask(x[i],resize)[0])
for i in range(len(test)):
    cv2.imwrite(test_path+filelist[i].split("/")[-1],x[i])
    cv2.imwrite( test_path_masked+ filelist[i].split("/")[-1], mask(x[i],resize)[0])

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
# unpickle("data/CIFAR10/cifar-10-batches-py/data_batch_1")