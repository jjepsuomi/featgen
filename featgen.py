#!/usr/bin/python

import pyximport; pyximport.install()
import numpy as np
import winfeat as wf
import lbp as LBP
import gabor as gb


# This function will take as input a 2D-image and produce a set of statistical and 
# Local binary pattern features into a text-file named 'feats.txt'
#
# INPUT: 
# 'data' 2D numpy array representing an image
# 'winSize' size of the window used, the size needs to be odd and > 1, e.g. 3,5,7,...
# 'stat' 0/1 boolean indicating whether windowed mean and standard deviation features are required
# 'lbp' 0/1 boolean indicating whether local binary pattern frequency features are required
# 'radius' distance between center point and lbp neighbours
# 'neighs' a positive divisible by 4 integer indicating the number of neighbours for lbp
# 'roinv' 0/1 boolean indicating if rotation invariant lbp matrix is required
# 'uniform' 0/1 boolean indicating if uniform lbp matrix is required
#
# OUTPUT: 
# 'feats.txt' a text file containing the calculated features. Each row represents a single pixel 
# in input image / matrix. A value NaN is given if the value cannot be calculated due to border limitations. 
#
# EXAMPLE USAGE: 
#
# Example 1:
#
# create_features(Some_data_matrix, 3, 1, 1, 1, 8, 1, 1)
# This is a function call to generate a set of features from input data-matrix 'Some_data_matrix'.
# We have 3x3 window size, statistical features are required, lbp features are required, lbp has radius 1, 
# 8 point neighbourhood selected in lbp, rotation-invarianvce is required, uniform patterns are required. 
#
# Example 2: 
# create_features(Some_data_matrix, 5, 0, 1, 2, 12, 1, 0)
# Same as in the preceding example, but window size is 5x5, statistical features are NOT required, 
# lbp features are required, lbp has radius 2,  12 point neighbourhood selected in lbp, rotation-invariance is required,
# uniform patterns are NOT required.


def create_features(fname, data, stat, winSize, gabor, gabor_freq, gabor_angles, gabor_sigX, gabor_sigY, lbp, radius, neighs, roinv, uniform):

    # Check that input data is in allowed form
    try:
        if winSize <= 1 or winSize % 2 != 1 or neighs <= 0 or neighs % 4 != 0 or not type(radius) == int or radius < 1:
            raise ValueError
        # Get data dimensions
        height = data.shape[0]
        width = data.shape[1]
        # Initialize output variable placeholders
        feature_counter, meanData, stdData, lbpData, lbpFreqData, gabors, outdata = 0, 0, 0, 0, 0, 0, 0
        
        if stat == 1:
            # Statistical features
            meanData, stdData = wf.get_stat_feats(data, winSize)
            feature_counter += 2
        if gabor == 1:
            gabors = []
            for j in range(0, gabor_angles):
                angle = (float(j) / gabor_angles) * np.pi
                gaborData = gb.get_gabor_feats(data, theta=angle, frequency=gabor_freq, sigmaX=gabor_sigX, sigmaY=gabor_sigY)
                gabors.append(gaborData)
                feature_counter += 1
        if lbp == 1:
            lbpdata = np.zeros(data.shape)
            net = np.zeros((4,3))
            dpoints = np.zeros((8))
            coords = circleCoords(neighs, radius)
            neighborhood = np.zeros((len(coords), 3))
            route = np.zeros((8,2), dtype = np.int16)
            # Local binary pattern histogram features 
            lbpData = LBP.lbp(data, lbpdata, neighs, radius, roinv, uniform, net, dpoints, coords, neighborhood, route)
            lbpFreqData, number_of_patterns = LBP.get_freqs(lbpData, winSize, neighs, roinv, uniform)
            feature_counter += number_of_patterns

        # Initialize dataholder
        outdata = np.zeros((height*width, feature_counter))
        outdata[:,0] = np.reshape(meanData, (height*width))
        outdata[:,1] = np.reshape(stdData, (height*width))
        for i in range(0, len(gabors)):
            outdata[:,2+i] = np.reshape(gabors[i], (height*width))
        n = len(gabors)+2
        outdata[:,n:] = np.reshape(lbpFreqData, (height*width, number_of_patterns))
        
        # Save data to textfile 
        np.savetxt(fname, outdata, fmt='%10.4f')
    except AttributeError, e:
        print "Given variable 'data' is not a numpy array!"
        print e
    except ValueError, e:
        print "Check the input parameters, window size must be > 1 and odd, radius must be positive integer >= 1, neighs must be divisible by 4!"
        print e
    else: 
        print "Features created succesfully"


def circleCoords(numPoints, radius):

    coords = np.zeros((numPoints, 2)) # Make a data-structure for the coordinates 
    coords[:,0] = np.round(radius*np.cos(2*np.pi*np.array(range(0,numPoints))/numPoints), 5)
    coords[:,1] = np.round(radius*np.sin(2*np.pi*np.array(range(0,numPoints))/numPoints), 5)
    return coords


