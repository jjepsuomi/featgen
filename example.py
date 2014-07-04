

import featgen as feat
import matplotlib.image as mpimg

# Get the data
data = mpimg.imread('pic.gif')
data = data[:,:,0] # To 2D

# Create the features
feat.create_features("test.txt", data.astype(float), winSize=3, stat=1, gabor=1, gabor_freq=0.33, gabor_angles=8, gabor_sigX=3, gabor_sigY=3, lbp=1, radius=1, neighs=8, roinv=1, uniform=1)


    