import matplotlib.pyplot as plt
import numpy as np

from numpy.fft import fftn, ifftn, fftshift, ifftshift
from skimage import io, color, transform


# make a  random seed 
np.random.seed(41)

# setting up the object 
Nx = 64
Ny = 64

# get the image from a url 
url = 'https://github.com/prickly-pythons/prickly-pythons/blob/master/code_from_meetings/research/cat.png?raw=true'
image =  io.imread(url)

x_true = color.rgb2gray(image)
x_true = transform.resize(x_true, (Nx,Ny))
x_true /= np.max(x_true)

fig = plt.figure(figsize=(8,6))
