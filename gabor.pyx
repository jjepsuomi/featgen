
from __future__ import print_function

import numpy as np
from scipy import ndimage as nd
from skimage.filter import gabor_kernel

def get_gabor_feats(data, theta = 0.785398163, frequency = 0.25, sigmaX = 3, sigmaY = 3):
    kernel = np.real(gabor_kernel(0.25, theta=theta, sigma_x=3, sigma_y=3))
    gabor_response = np.sqrt(nd.convolve(data, np.real(kernel), mode='wrap')**2 + nd.convolve(data, np.imag(kernel), mode='wrap')**2)
    return gabor_response



