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
url = 'https://www.safetysign.com/images/source/page-list-pages/productgrid-n-t/Y1249.png'
# url = 'media/src/traffic.png'
image =  io.imread(url)

x_true = color.rgba2rgb(image)
x_true = color.rgb2gray(x_true)
x_true = transform.resize(x_true, (Nx,Ny))
x_true /= np.max(x_true)


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
im = ax.imshow(x_true, interpolation='nearest')
ax.set_title('Input')



def get_image_from_url(link):
    # setting up the object 
    Nx = 64
    Ny = 64

    
    image =  io.imread(link)

    x_true = color.rgba2rgb(image)
    x_true = color.rgb2gray(x_true)
    x_true = transform.resize(x_true, (Nx,Ny))
    x_true /= np.max(x_true)
    return x_true

def plot(image, title : str, ax_int : int):
    """ image is used as x_true in the example below
            im = ax.imshow(x_true, interpolation='nearest')
        ax_int is used as 111
            ax = fig.add_subplot(111)
     """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(ax_int)
    im = ax.imshow(image, interpolation='nearest')
    ax.set_title(title)

# Projection operators 
def P_S(x, S_in):
    x_new = x*S_in['supp']
    return x_new

def P_M(x, M_in):
    X = fftn(x)
    X_new = X/np.abs(X) * M_in['M_data'] # X_new = M_in['M_data'] * np.angle(X))
    x_new = ifftn(X_new)
    return x_new

# The difference Map
def R_M(x, gamma_M, M_in):
    return (1+gamma_M)*P_M(x,M_in) - gamma_M*x

def R_S(x, gamma_S, S_in):
    return (1+gamma_S)*P_M(x,S_in) - gamma_S*x

def DM(x, beta, gamma_S, gamma_M, M_in, S_in):
    x_PMRS = P_M(R_S(x, gamma_S, S_in), M_in)
    x_PSRM = P_S(R_M(x, gamma_M, M_in), S_in)

    x_new = x + beta*(x_PMRS-x_PSRM)

    return x_new, x_PSRM
    
def convolution_filter(x, kernel):
    return ifftn(x)*kernel

# taking the fourier transform of the image and clculating the fourier magnitude 
X_true = fftn(x_true)
M_true = np.abs(X_true)

# making an initial support 
supp = np.zeros([Nx,Ny])
supp[16:48,16:48] = 1

# display the magnitude and the support 
fig = plt.figure(figsize=(16,6))

ax = fig.add_subplot(121)
im = ax.imshow(supp)
plt.colorbar(im)
ax.set_title('Initial Support')

ax = fig.add_subplot(122)
im = ax.imshow(fftshift(np.log10(M_true)))
plt.colorbar(im)
ax.set_title('Fourier Magnitude data')


# Shrink Wrap 
C_lp = np.zeros([Nx,Ny])
C_lp[20:44,20:44] = 1
C_lp = ifftshift(C_lp)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
im = ax.imshow(fftshift(C_lp))
ax.set_title("fftshift'ed convolution kernel")
plt.colorbar(im)





x_lp = convolution_filter(x_true, C_lp)
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(221)
im = ax.imshow(x_true, clim=[0,1])
plt.colorbar(im)
ax.set_title('x_true')

ax = fig.add_subplot(222)
im = ax.imshow(np.abs(x_lp), clim=[0,1])
plt.colorbar(im)
ax.set_title('Initit')
plt.show()