The following are the functions of each of the scripts:

data.py - Tool for producing custom dataset from original dataset (cropping, resizing, image mask creation)
ImageDataset.py - Dataset object for the flowers dataset in this project. 
model.py - contain the neural nets used in deep inpainting
train.py - code to train the model (Pytorch)
test.py - code to produce results from the test set

We need to first run data.py to generate data, then train.py to train, and finally test.py to get unpainted images.
