# Special Topics - ECE 491

## Introduction
The main goal of this research is to recover signals in the presence of Speckle Noise. The way this can be done is using Machine Learning, Autoencoders, and algorithms based off that. The code I have will be outlined below. 

## Problem 
Speckle noise is defined as "granular noise texture degrading the quality as a consequence of interference among wavefronts in coherent imaging systems". This essentially means that it is something that is degrading the image quality. Here is an example: 

<img src="https://github.com/akhilvreddy/ECE491/blob/main/2-Figure1-1.png" alt="Speckle Noise Image" class="right">

The goal here is to get from an image that looks like the one on the right (which is what we currently get in SAR and OCT imaging) to something like the image on the left. 

In essence, we are trying to remove the granular part of the image for better resolution.  

### Graphical Depiction
In this case, we are talking about images and images having speckle noise, but I would first like to show speckle noise's effect on signals. Since images can be represented as vectors, this would be a good way to visualize what is happening. Since I am going to be converting the image we are working with to patches and then later to a vector, it is good to see what changes happen to a single vector. 

The noise formula is modeled by the following: $$\textbf{y} = AX_o\textbf{w} + \textbf{z}$$

Here is what all the variables are: 
- $\textbf{y}$ : This is the final measuremetn of the signal we end up with, and is the one that we see. 
- $A$ : This is a multiplicative constant (can be in the form of a matrix)/
- $X_o$ : This is the original signal in the form of a matrix. The signal elements are on the diagonal of a square matrix.
- $\textbf{w}$ : The speckle noise, this is the main issue we are dealing with. 
- $\textbf{z}$ : This is the white guasian additive noise. 

We can generate random values for speckle noise, additive white gaussian noise, and the original signal to see a sample comparison between $X_o$ and $y$. 

We can take $w$ as some random multiplicative values, ranging between 0.8-1.2 because we don't want too much difference, but just enough to see. 
We can take $X_o$ as the matrix of the array [5.4, 7.65, 9.4, 3.4] as spread along its diagonal. This was randomly generated.
We can take $z$ as a guassian random distributed noise:
Ignoring $A$ for now, we can just set it equal to 1.

We end up with the following: 



### Turning an Image into a vector & image patches
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


## Next Steps

Professor Jalali's paper is mainly a theoretical framework of what I want to implement in code. The vector operations and such are the things I want to physically impelment by using actual images. The way I am doing is this by the following: 

### Recovery Algorithms

One of the ways to get the signal (or image in this case) back from speckle noise is by Projected Gradient Descent. The cost function in this case would be 


### Autoencoders

Autoencoders are the biggest tools that allow us to solve inverse problems. The way we are going to solve the vector equation is by trying to inverse it, kind of like an algebraic equation, but we cannot do the same elementary operations for a vector equation involving matrices. 

```
```

### Quetions to ask Professor: 

- How to implement the algorithm 1 from page 13. 
- How to properly input an image as a vector so we can "add" speckle noise to it.
- How to create "speckle noise" as a means to test. I want to get an image and show how it changes with the effect of speckle noise.


- how to make sure my autoencoder is working well?



#Progress (10/3)
- first part is good
- need to fix the image patch part (going back to image)
- how to check if my autoencoder is working proplerly and steps after that 
- what to do after that? implementing the algorithms in code
