# Special Topics - ECE 491

## Introduction
The main goal of this research is to recover signals in the presence of Speckle Noise. The way this can be done is using Machine Learning, Autoencoders, and algorithms based off that. The code I have will be outlined below. 

## Problem 
Speckle noise is defined as "granular noise texture degrading the quality as a consequence of interference among wavefronts in coherent imaging systems". This essentially means that it is something that is degrading the image quality. Here is an example: 

<img src="https://d3i71xaburhd42.cloudfront.net/e7160408b2f36f582caf9c6e1714f98415126469/2-Figure1-1.png" alt="Speckle Noise Image" class="center">

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


## Next Steps

Professor Jalali's paper is mainly a theoretical framework of what I want to implement in code. The vector operations and such are the things I want to physically impelment by using actual images. The way I am doing is this by the following: 

### Recovery Algorithms

One of the ways to get the signal (or image in this case) back from speckle noise is by Projected Gradient Descent. The cost function in this case would be 


### Quetions to ask Professor: 

- How to implement the algorithm 1 from page 13. 
- How to properly input an image as a vector so we can "add" speckle noise to it.
- How to create "speckle noise" as a means to test. I want to get an image and show how it changes with the effect of speckle noise.


- how to make sure my autoencoder is working well?
