# Special Topics - ECE 491

## Introduction
The main goal of this research is to recover signals in the presence of Speckle Noise. The way this can be done is using Machine Learning, Autoencoders, and algorithms based off that. The code I have will be outlined below. 

## Problem 
Speckle noise is defined as "granular noise texture degrading the quality as a consequence of interference among wavefronts in coherent imaging systems". This essentially means that it is something that is degrading the image quality. Here is an example: 

<p align="center">
  <img 
    width="504"
    height="228"
    src="https://github.com/akhilvreddy/ECE491/blob/main/2-Figure1-1.png"
  >
</p>

The goal here is to get from an image that looks like the one on the right (which is what we currently get in SAR and OCT imaging) to something like the image on the left. 

In essence, we are trying to remove the granular part of the image for better resolution.  

## Graphical Depiction
In this case, we are talking about images and images having speckle noise, but I would first like to show speckle noise's effect on signals. Since images can be represented as vectors, this would be a good way to visualize what is happening. Since I am going to be converting the image we are working with to patches and then later to a vector, it is good to see what changes happen to a single vector. 

The noise formula is modeled by the following: $$\textbf{y} = AX_o\textbf{w} + \textbf{z}$$

Here is what all the variables are: 
- $\textbf{y}$ : This is the final measuremetn of the signal we end up with, and is the one that we see. 
- $A$ : This is a multiplicative constant (can be in the form of a matrix)/
- $X_o$ : This is the original signal in the form of a matrix. The signal elements are on the diagonal of a square matrix.
- $\textbf{w}$ : The speckle noise, this is the main issue we are dealing with. 
- $\textbf{z}$ : This is the white guasian additive noise. 

### Test vector
We can generate random values for speckle noise, additive white gaussian noise, and the original signal to see a sample comparison between $X_o$ and $y$. 

We can take $w$ as some random multiplicative values, ranging between 0.8-1.2 because we don't want too much difference, but just enough to see. 
We can take $X_o$ as the matrix of the array [5.4, 7.65, 9.4, 3.4] as spread along its diagonal. This was randomly generated.
We can take $z$ as a guassian random distributed noise:
Ignoring $A$ for now, we can just set it equal to 1.

* Here, I am converting all of the lists to python vectors so that we can do operations on them.

We end up with the following: 
```
import numpy as np

list1 = []
list2 = []
```


### Test image (converting image to vector using image patches)
The images we are working are pretty big and high in quality so we take patches of them in order to do the analysis. This is how we take the patches in python, using the _scikit learn_ library: 
```
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image

sample_image = load_sample_image("test1.jpg")
print('Image shape: {}'.format(sample_image.shape)) 
```
> "test1.jpg" was imported from desktop, in this case it was the complete image

For one of the images I was dealing with, after running this script, I ended up with the following: 

[[[174 201 231],[174 201 231]],[[173 200 230],[173 200 230]]]

These numbers correspond to the ___ of the image. 

*QUESTION*

how to go from the vector back to the image itself?

I also consulted the following video: https://www.youtube.com/watch?v=7IL7LKSLb9I.


## Getting to our result

Professor Jalali's paper is mainly a theoretical framework of what I want to implement in code. The vector operations and such are the things I want to physically impelment by using actual images. The way I am doing is this by the following: 

### Autoencoders

Autoencoders are the biggest tools that allow us to solve inverse problems. The way we are going to solve the vector equation is by trying to inverse it, kind of like an algebraic equation, but we cannot do the same elementary operations for a vector equation involving matrices. 

```
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import os
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"]='0'
ids=[0]
torch.cuda.empty_cache()
#initial values
EPOCH = 20 #this can give a good result
BATCH_SIZE = 64
LR = 0.001

# load images data
train_data = torchvision.datasets.ImageFolder('./Training_set', transform=transforms.Compose([
transforms.ToTensor()]))
train_loader=Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class UNet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UNet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

unet=UNet(in_ch=3,out_ch=3).cuda()
optimizer=torch.optim.Adam(unet.parameters(),lr=LR) #optimizer: SGD, Momentum, Adagrad, etc. This one is better.
loss_func=nn.MSELoss() #loss function: MSE

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # two layers encoder
            nn.Linear(64 * 64*3, 12000),
            nn.Sigmoid(),  # ReLU, Tanh, etc.
             # input is in (0,1), so select this one
            nn.Linear(12000, 3000),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(  # two layers decoder
            nn.Linear(3000, 12000),
            nn.Sigmoid(),
            nn.Linear(12000, 64 * 64*3),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#Load 4 NNs
autoencoder=Autoencoder().cuda()
autoencoder.load_state_dict(torch.load('AE-trained.pkl')) #load the parameter values of NN

for epoch in range(EPOCH):
    for step, (x,x_label) in enumerate(train_loader): #train_loader has the number of batches, data, and label
        b_y=x.cuda()
        batch_size = x.size()[0]
        b_x=x.view(-1,64*64*3).cuda()
        decoded=autoencoder(b_x)
        decoded=decoded.view(batch_size,3,64,64)
        decoded=decoded.cuda()
        #running in the neural network
        output=unet(decoded)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()#initialize the optimizer
        loss.backward()
        optimizer.step()

        if step%40==0:
            print('Epoch:',epoch, '| tran loss : %.4f' % loss.data.cpu().numpy())

torch.save(unet.state_dict(),'Unet-trained.pkl'
```


### Recovery Algorithms

One of the ways to get the signal (or image in this case) back from speckle noise is by Projected Gradient Descent. The cost function in this case would be 

#### Recovery using Generative Functions (GFs)

Before getting into other more complicated algorithms, I want to go over others. Recovery using GFs is used for 


#### Projected Gradient Descent (PGD)

This is one of the ways we reduce the cost function step-by-step to drive closer to the solution everytime. Understading the pseudo-code is very important before coding it up. 

* We are assuming that the result is **x<sub>t</sub>**.

```
Xo = diag(xo)    //initalize the matrix
Bo = A(Xo)^2(A^T)

for t = 1:T do 
  for i = 1:n do
  
  s(t, i) = somethingsomething
  
  end 
  
  x_t = pi*c_r*s_t
  Xt = diag(xt)
  Bt = A(X_t)^2(A^T)
end
```

Let's unpack this alogrithm. The first two lines should be pretty self-explanatory - we are just setting up the basic matrix and constants in order to do the first iteration calcuations.

The nested for loops is where we get into the meat of the algorithm. For the inner-most loop, we are trying to find values of s_(t,i). Since t stays the same for a single loop, we get n specific values of the s vector. 

After exiting that loop, we set all the values of the x_t vector to a specific constant times those values. 

The X_t matrix gets updated from this everytime. 

By running through both of these for-loops we are inching closer towards an answer everytime. The main line that is getting us to reduce our cost-function is: 

<p align="center">
  <img 
    width="708"
    height="78"
    src="https://github.com/akhilvreddy/ECE491/blob/main/pic2.png"
  >
</p>


### Simulating our Results

#### Mean Squared Error (MSE) vs. n



## Analysis of Results



### Future ideas/work




### Quetions to ask Professor: 

- How to properly input an image as a vector so we can "add" speckle noise to it.
- How to create "speckle noise" as a means to test. I want to get an image and show how it changes with the effect of speckle noise.


- how to make sure my autoencoder is working well?



# Progress (10/3)
- first part is good
- need to fix the image patch part (going back to image)
- how to check if my autoencoder is working proplerly and steps after that 
- how to define autoencoder
- what to do after that? implementing the algorithms in code
