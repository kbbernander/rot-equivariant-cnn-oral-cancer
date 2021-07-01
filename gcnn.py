#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:14:45 2020

@author: karl
"""

import torch
import matplotlib.pyplot as pyplot
import pickle

from e2cnn import gspaces
from e2cnn import nn

print("E2CNN, VGG16 simplified version")

class GCNN(torch.nn.Module):
    
    def __init__(self, n_classes=2):
        	
        super(GCNN, self).__init__()
        
        # Trivial representations for ordinary convolutions
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 16 feature fields, each transforming under the regular representation of C4
        out_type = nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size=2, stride=2, padding=0)
        )
        
        # convolution 3
        in_type = self.block2.out_type
        out_type = nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 4
        in_type = self.block3.out_type
        out_type = nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPool(out_type, kernel_size=2, stride=2, padding=0)
        )
        
        # convolution 5
        in_type = self.block4.out_type
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 6
        in_type = self.block5.out_type
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 7
        in_type = self.block6.out_type
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size=2, stride=2, padding=0)
        )
        
        # convolution 8
        in_type = self.block7.out_type
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block8 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 9
        in_type = self.block8.out_type
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block9 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # convolution 10
        in_type = self.block9.out_type
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block10 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool4 = nn.SequentialModule(
           nn.PointwiseMaxPool(out_type, kernel_size=2, stride=2, padding=0)
       )
        
        # convolution 11
        in_type = self.block10.out_type
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block11 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 12
        in_type = self.block11.out_type
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block12 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 13
        in_type = self.block12.out_type
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block13 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=4, padding=0, maximum_offset=0),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool5 = nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size=2, stride=1, padding=0)
        )
        in_type = self.block13.out_type
        self.gpool = nn.GroupPooling(in_type)
        
        # number of output channels
        c = self.gpool.out_type.size
        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, n_classes),
        )
    
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.pool3(x)

        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.pool4(x)

        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)

        x = self.pool5(x)
        
        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        return x
    
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose

import torchvision.transforms.functional as TF
import random


import numpy as np

from PIL import Image

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    print ("cuda to the rescue")
else:
    print ("no cuda found :(")

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class MnistRotDataset(Dataset): #For rotated MNIST
    
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32) 
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)

totensor = ToTensor()

class OralDataset(Dataset): #For the oral cancer dataset
    
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = "../data/data/cells_training_4800.amat"
        else:
            file = "../data/data/cells_test_full.amat"
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 80, 80).astype(np.float32) 
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)

totensor = ToTensor()

model = GCNN().to(device)

def test_model(model: torch.nn.Module, x: Image):
    # evaluate the `model` on 4 rotated versions of the input image `x`
    model.eval()
    
    print()
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(4):
            x_transformed = totensor(x.rotate(r*90., Image.BILINEAR)).reshape(1, 1, 80, 80)
            x_transformed = x_transformed.to(device)

            y = model(x_transformed)
            #y = y.to('cpu').numpy().squeeze()
            
            angle = r * 90
            print("{:5d} : {}".format(angle, y))
    print('##########################################################################################')
    print()

# build the test set    
#mnist_test = MnistRotDataset(mode='test')
oral_test = OralDataset(mode='test')

# retrieve the first image from the test set
x, y = next(iter(oral_test))

#print(model)

# evaluate the model
test_model(model, x)

#the rotation transform isn't used here. Instead, we augment the images beforehand and load them with the dataloader.
rotation_transform = MyRotationTransform(angles=[0, 90, 180, 270])

train_transform = Compose([
    #rotation_transform,
    totensor,
])

#mnist_train = MnistRotDataset(mode='train', transform=train_transform)
oral_train = OralDataset(mode='train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(oral_train, batch_size=128, shuffle=True)


test_transform = Compose([
    totensor,
])
#mnist_test = MnistRotDataset(mode='test', transform=test_transform)
oral_test = OralDataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(oral_test, batch_size=128, shuffle=True)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=256.0)

#print(model)


#pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(pytorch_total_params)


nr_epochs = 200
nr_epochs_vector = 40
xb = np.linspace(1 , nr_epochs_vector, nr_epochs_vector)
yb = np.linspace(1 , nr_epochs_vector, nr_epochs_vector)
i_best=0
for epoch in range(nr_epochs):
    model.train()
    for i, (x, t) in enumerate(train_loader):

        optimizer.zero_grad()

        x = x.to(device)
        t = t.to(device)

        y = model(x)

        loss = loss_function(y, t)

        loss.backward()

        optimizer.step()
    
    if epoch % 5 == 0:
        confusion_matrix = torch.zeros(2, 2)
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(test_loader):

                x = x.to(device)
                t = t.to(device)
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
                
                for a, b in zip(t.view(-1), prediction.view(-1)):
                    confusion_matrix[a.long(), b.long()] += 1
        print(f"epoch {epoch} | test accuracy: {correct/total*100.}")
        print(f"sensitivity: {confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])}")
        print(f"specificity: {confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])}")
        print(confusion_matrix)
        yb[int(epoch/5)]=(correct/total*100)
        if((correct/total*100)>i_best):
            i_best = correct/total*100
            print(f"this is currently the best epoch {i_best}")
        confusion_matrix = torch.zeros(2, 2)
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(train_loader):

                x = x.to(device)
                t = t.to(device)
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
                
                for a, b in zip(t.view(-1), prediction.view(-1)):
                    confusion_matrix[a.long(), b.long()] += 1
        print(f"epoch {epoch} | train accuracy: {correct/total*100.}")
        print(f"sensitivity: {confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])}")
        print(f"specificity: {confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])}")
        print(confusion_matrix)
        xb[int(epoch/5)]=(correct/total*100)


# build the test set    
#other_mnist_test = MnistRotDataset(mode='test')
other_oral_test = OralDataset(mode='test')

# retrieve the first image from the test set
x, y = next(iter(other_oral_test))


# evaluate the model
test_model(model, x)

### Below is for plotting results ###
#epochs = np.linspace(0 , 195, 39)

#pyplot.figure(1)
#pyplot.suptitle('GCNN training accuracy, no augmentations, VGG16_full, learning rate 10-5', fontsize = 12)
#pyplot.xlabel('epochs')
#pyplot.ylabel('accuracy')
#pyplot.axis([0, 195, 0, 100])
#pyplot.plot(epochs,xb[0:39], label = '4800 examples') 
#pyplot.plot(epochs,xb600, label = '600 examples') 
#pyplot.plot(epochs,xb1200, label = '1200 examples') 
#pyplot.plot(epochs,xb2400, label = '2400 examples') 
#pyplot.plot(epochs,xb4800, label = '4800 examples') 
#pyplot.plot(epochs,xb8500, label = '8508 examples') 
#pyplot.legend()
#pyplot.savefig('13_noaug_4800_VGG_GCNN_l2_256_00_training.png')

#pyplot.figure(2)
#pyplot.suptitle('GCNN testing accuracy, no augmentations, VGG16_full, learning rate 10-5', fontsize = 12)
#pyplot.xlabel('epochs')
#pyplot.ylabel('accuracy')
#pyplot.axis([0, 195, 0, 100])
#pyplot.plot(epochs,yb[0:39], label = '4800 examples') 
#pyplot.plot(epochs,xb600, label = '600 examples') 
#pyplot.plot(epochs,xb1200, label = '1200 examples') 
#pyplot.plot(epochs,xb2400, label = '2400 examples') 
#pyplot.plot(epochs,xb4800, label = '4800 examples') 
#pyplot.plot(epochs,xb8500, label = '8508 examples') 
#pyplot.legend()
#pyplot.savefig('13_noaug_4800_VGG_GCNN_l2_256_00_testing.png')

### Saving data to file ###
#with open('13_noaug_4800_VGG_GCNN_200epochs_l2_256_00.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump([xb, yb], f)
#print
