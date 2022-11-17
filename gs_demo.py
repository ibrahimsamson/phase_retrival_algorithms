import math
import numpy as np


def Ger_Sax_algo(img, max_iter):
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    pm_f = np.ones((h, w))
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))

    signal_s = am_s*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f = np.fft.fft2(signal_s)
        pm_f = np.angle(signal_f)
        signal_f = am_f*np.exp(pm_f * 1j)
        signal_s = np.fft.ifft2(signal_f)
        pm_s = np.angle(signal_s)
        signal_s = am_s*np.exp(pm_s * 1j)

    pm =pm_f
    return pm


import cv2
from phase_retrieval import *
import matplotlib.pyplot as plt
import numpy as np

filename = 'images.jpg'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
img = img.astype(float)
img = np.asarray(img, float)
max_iters = 1000
phase_mask = Ger_Sax_algo(img, max_iters)
plt.figure(1)
plt.subplot(131)
plt.imshow(img)
plt.title('Desired image')
plt.subplot(132)
plt.imshow(phase_mask)
plt.title('Phase mask')
plt.subplot(133)
recovery = np.fft.ifft2(np.exp(phase_mask * 1j))
plt.imshow(np.absolute(recovery)**2)
plt.title('Recovered image')
plt.show()

