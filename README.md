# Special Topics - ECE 491

## Introduction
The main goal of this research is to recover signals in the presence of Speckle Noise. The way this can be done is using Machine Learning, Autoencoders, and algorithms based off that. The code I have will be outlined below. 

# Problem 
Speckle noise is defined as "granular noise texture degrading the quality as a consequence of interference among wavefronts in coherent imaging systems". This essentially means that it is something that is degrading the image quality. Here is an example: 

<img src="https://d3i71xaburhd42.cloudfront.net/e7160408b2f36f582caf9c6e1714f98415126469/2-Figure1-1.png" alt="Speckle Noise Image" class="center">

The goal here is to get from an image that looks like the one on the right (which is what we currently get in SAR and OCT imaging) to something like the image on the left. 

In essence, we are trying to remove the granular part of the image for better resolution.  

