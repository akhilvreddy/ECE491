# Special Topics - ECE 491

## Introduction
The main goal of this research is to recover signals in the presence of Speckle Noise. The way this can be done is using Machine Learning, Autoencoders, and algorithms based off that. The code I have will be outlined below. 

## Problem 
Speckle noise is defined as "granular noise texture degrading the quality as a consequence of interference among wavefronts in coherent imaging systems". This essentially means that it is something that is degrading the image quality. Here is an example: 

<img src="https://d3i71xaburhd42.cloudfront.net/e7160408b2f36f582caf9c6e1714f98415126469/2-Figure1-1.png" alt="Speckle Noise Image" class="center">

The goal here is to get from an image that looks like the one on the right (which is what we currently get in SAR and OCT imaging) to something like the image on the left. 

In essence, we are trying to remove the granular part of the image for better resolution.  

## Next Steps

Professor Jalali's paper is mainly a theoretical framework of what I want to implement in code. The vector operations and such are the things I want to physically impelment by using actual images. The way I am doing is this by the following: 

### Recovery Algorithms

One of the ways to get the signal (or image in this case) back from speckle noise is by Projected Gradient Descent. The cost function in this case would be 


### Quetions to ask Professor: 

- How to implement the algorithm 1 from page 13. 
- How to properly input an image as a vector so we can "add" speckle noise to it.
- How to create "speckle noise" as a means to test. I want to get an image and show how it changes with the effect of speckle noise.
