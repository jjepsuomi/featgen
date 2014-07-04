#!/usr/bin/python

import numpy as np
from scipy.ndimage import generic_filter

# This function will return the statistical features
# 
#
# INPUTS: 
# 'data' the data from which statistical features are to be calculated
# "winSize" specifying the window size, must be odd and > 1
#
# OUTPUT: 
# 'meanData, stdData' statistical feature matrices (numpy ndarrays)

def get_stat_feats(data, winSize):
    data = data.astype(float)
    mean = lambda x: x.mean()
    std = lambda x: x.std()

    meanData = generic_filter(data, mean, size=winSize)
    stdData = generic_filter(data, std, size=winSize)

    return np.round(meanData,2), np.round(stdData, 2)


                            
def tofile(data, fo):
    
    for row in range(0, data.shape[0]):
        if len(data.shape) > 1:
            for col in range(0, data.shape[1]):
                fo.write(str(data[row, col]))
                fo.write(" ")
            fo.write("\n")
        else: 
            fo.write(str(data[row]))
            fo.write(" ")
    