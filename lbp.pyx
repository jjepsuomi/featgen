
import numpy as np
cimport numpy as np
import math
from scipy.ndimage import generic_filter

DTYPE = np.double
ctypedef np.double_t DTYPE_t



# Will return the coordinates for the neighbourhood points of center pixel, but in an inverse
# order compared to circleCoords.
# [0, 0] is assumed to be the center pixel 
#
# INPUTS:
# 'points' is a positive integer indicating the number of neighbours 
# 'radius' is the distance between center pixel and the neighbours 
#
# OUTPUT: 
# 'coords' 2D-numpy array indicating the X- and Y-coordinates of the neighbourhood

def circleCoordsD(numPoints, radius):

    coords = np.zeros((numPoints, 2)) # Make a data-structure for the coordinates 
    coords[:,0] = np.round(radius*np.cos(2*np.pi*np.array(range(numPoints-1,-1,-1))/numPoints), 5)
    coords[:,1] = np.round(radius*np.sin(2*np.pi*np.array(range(numPoints-1,-1,-1))/numPoints), 5)
    return coords

    
# Will return the coordinates and values of the origin's neighbourhood, does bilinear interpolation
# [0, 0] is assumed to be the center pixel 
#
# INPUTS:
# 'dp' is a group of neighbouring pixels 
# 'co' is the coordinates for the neighbours of center pixel
# 'thres' is the value of center point, i.e. [0,0]
# 'r' is the radius, i.e. distance between center point and its neighbours
#
# OUTPUT:
# 'neighborhood' 2D-numpy array indicating the X- and Y-coordinates and the value of the neighbourhood

def binterp(double [:] dp, double [:, :] co, double thres, double r, double [:, :] net, double [:, :] neighborhood):
    
    cdef double X, Y, fR1, fR2, fP
    cdef int ind, i
    #neighborhood = np.zeros((len(co), 3)) # Make a data structure for the coordinates and values
    ind = 0 # Index used for locating the 0, 90, 180 and 270 degrees pixels 
    #net = np.zeros((4,3)) # Net consisting of four pixels, these pixels are used for bilinear interpolation
    for i in range(0, len(co)): # Calculate values for all the coordinates
        # Set the neighbourhood coordinates
        X = co[i,0]
        Y = co[i,1]
        neighborhood[i,0] = X
        neighborhood[i,1] = Y    
        # If we are at 0, 90, 180 or 270 degrees pixels, we already have these values so interpolation is not needed
        if (X == r and Y == 0) or (X == 0 and Y == r) or (X == -r and Y == 0) or (X == 0 and Y == -r):
            neighborhood[i,2] = dp[ind]
            ind += 2  
        # We need bilinear interpolation 
        else: # Depending on where the coordinates are, we need different net       
            if (X > 0 and Y > 0): # Top-right               
                net[0,0] = 0;
                net[0,1] = 0;
                net[0,2] = thres
                net[1,0] = r;
                net[1,1] = 0;
                net[1,2] = dp[0]
                net[2,0] = r;
                net[2,1] = r;
                net[2,2] = dp[1]
                net[3,0] = 0;
                net[3,1] = r;
                net[3,2] = dp[2]
            elif (X < 0 and Y > 0): # Top-left
                net[0,0] = -r; net[0,1] = 0; net[0,2] = dp[4]
                net[1,0] = 0; net[1,1] = 0; net[1,2] = thres
                net[2,0] = 0; net[2,1] = r; net[2,2] = dp[2]
                net[3,0] = -r; net[3,1] = r; net[3,2] = dp[3]
            elif (X < 0 and Y < 0): # Lower-left
                net[0,0] = -r; net[0,1] = -r; net[0,2] = dp[5]
                net[1,0] = 0; net[1,1] = -r; net[1,2] = dp[6]
                net[2,0] = 0; net[2,1] = 0; net[2,2] = thres
                net[3,0] = -r; net[3,1] = 0; net[3,2] = dp[4]
            elif (X > 0 and Y < 0): # Lower-right
                net[0,0] = 0; net[0,1] = -r; net[0,2] = dp[6]
                net[1,0] = r; net[1,1] = -r; net[1,2] = dp[7]
                net[2,0] = r; net[2,1] = 0; net[2,2] = dp[0]
                net[3,0] = 0; net[3,1] = 0; net[3,2] = thres
            # Do bilinear interpolation, see wikipedia         
            fR1 = ((net[1,0]-X)/(net[1,0]-net[0,0]))*net[0,2] + ((X-net[0,0])/(net[1,0]-net[0,0]))*net[1,2]
            fR2 = ((net[1,0]-X)/(net[1,0]-net[0,0]))*net[3,2] + ((X-net[0,0])/(net[1,0]-net[0,0]))*net[2,2]
            fP = ((net[2,1]-Y)/(net[2,1]-net[0,1]))*fR1 + ((Y-net[0,1])/(net[2,1]-net[0,1]))*fR2
            neighborhood[i,2] = fP
            
    return neighborhood


# Will return the corresponding decimal number of binary vector
#
# INPUTS:
# 'binvec' is a binary vector of type numpy ndarray 
#
# OUTPUT:
# 'value' The evaluated decimal value of the binary vector 

def evaluate(binvec):
    
    return int(np.dot(np.reshape(binvec, (1, len(binvec))), 2**np.arange(len(binvec)))[0])


# Rotates a given binary vector one step to the right 
#
# INPUTS:
# 'binvec' is a binary vector of type numpy ndarray 
#
# OUTPUT:
# 'temp' The right-rotated binary vector

def rotateDown(binvec):
    
    temp = np.zeros((len(binvec),1))
    temp[0:len(temp)-1] = binvec[1:]
    temp[len(temp)-1] = binvec[0]
    return temp 


# Rotates a given binary vector one step to the left 
#
# INPUTS:
# 'binvec' is a binary vector of type numpy ndarray 
#
# OUTPUT:
# 'temp' The left-rotated binary vector

def rotateUp(binvec):
    
    temp = np.zeros((len(binvec),1))
    temp[1:len(temp)] = binvec[0:len(binvec)-1]
    temp[0] = binvec[len(binvec)-1]
    return temp
    

# Will check whether a given binary vector is uniform or not 
# A binary pattern is uniform if when rotated one step, the number of
# bit values changing values is <= 2
#
# INPUTS:
# 'binvec' is a binary vector of type numpy ndarray 
#
# OUTPUT:
# 'True/False' boolean indicating uniformness

def isUniform(binvec):

    temp = rotateDown(binvec) # This will rotate the binary vector one step down
    if (np.count_nonzero(binvec!=temp)) > 2:
        return False
    else:
        return True
    
# Will return a rotation invariant binary vector
#
# INPUTS:
# 'binvec' is a binary vector of type numpy ndarray 
#
# OUTPUT:
# 'binvec' rotation invariant version    

def roInv(binvec):
    
    minFound = 0
    while minFound != 1:
        cvalue = evaluate(binvec)
        rotd = rotateDown(binvec)
        rotu = rotateUp(binvec)
        dvalue = evaluate(rotd)
        uvalue = evaluate(rotu)
        if dvalue < cvalue:
            binvec = rotd
        elif uvalue < cvalue:
            binvec = rotu
        else:
            minFound = 1
    return binvec
            

# Will return a binary vector presentation of the neighbourhood
#
# INPUTS:
# 'ndata' numpy ndarray consisting of the neighbourhood values 
# 'thres' decimal value indicating the value of the center pixel
#
# OUTPUT:
# 'bvec' binary vector presentation of the neighbourhood 

def toBinvec(ndata, thres):
    ndata = np.asarray(ndata)
    return  np.where(ndata[:,2] < thres, 0, 1 ).reshape(-1,1)


# Will return the lbp-code of a binary vector. Rotation invariance and uniformness is checked
#
# INPUTS:
# 'binvec' binary vector indicating the neighbourhood 
# 'roinv' 0/1 boolean indicating whether rotation invariant lbp-code is required 
# 'uniform' 0/1 boolean indicating if uniform lbp-code is required
#
# OUTPUT:
# '-1 or evaluate' decimal lbp-code value

cdef int lbpCode(binvec, int roinv, int uniform):

    if roinv == 1: # Rotation invariance required 
        binvec = roInv(binvec)
    if uniform == 1 and not isUniform(binvec):
        return -1
    else:
        return evaluate(binvec)
    

# Will return a local binary pattern transformation matrix from the input data matrix
#
# INPUTS:
# 'data' is the data to be transformed  
# 'npoints' positive and even integer indicating the number of neighbours  
# 'radius' positive integer indicating the radius i.e. distance between center pixel and its neighbours
# 'roinv' 0/1 boolean indicating if rotation invariant lbp matrix is required
# 'uniform' 0/1 boolean indicating if uniform lbp matrix is required
# 
# OUTPUT:
# 'lbpdata' a matrix containing lbp-codes in the pixels, i.e. lbp-transformation of the input image

def lbp(double [:, :] data, double [:, :] lbpdata, int npoints, int radius, int roinv, int uniform, double [:, :] net, double [:] dpoints, double [:, :] coords, double [:, :] neighborhood, short [:, :] route):
    
    cdef np.ndarray bvec
    cdef int rows, cols, row, col, err, i, indCol, indRow
    
    if npoints > 0 and npoints % 4 == 0:
        #lbpdata = np.zeros(data.shape)
        #dpoints = np.zeros((8, 1))
        #route = np.zeros((8,2))
        route[0,0] = 1; route[0,1] = 0
        route[1,0] = 1; route[1,1] = 1
        route[2,0] = 0; route[2,1] = 1
        route[3,0] = -1; route[3,1] = 1
        route[4,0] = -1; route[4,1] = 0
        route[5,0] = -1; route[5,1] = -1
        route[6,0] = 0; route[6,1] = -1
        route[7,0] = 1; route[7,1] =-1
        rows = data.shape[0]
        cols = data.shape[1]
        for row in range(0, rows):
            for col in range(0, cols):
                thres = data[row, col]
                err = 0
                for i in range(0,len(dpoints)):
                    indCol = col + route[i,0]*radius
                    indRow = row + route[i,1]*radius
                    if indRow >= 0 and indRow <= rows-1 and indCol >= 0 and indCol <= cols-1 and err == 0:
                        dpoints[i] = data[indRow, indCol]
                    else:
                        err = 1
                if err != 1:
                    values = binterp(dpoints, coords, thres, radius, net, neighborhood)
                    bvec = toBinvec(values, thres)
                    lbpdata[row, col] = lbpCode(bvec, roinv, uniform)
                else: 
                    lbpdata[row, col] = np.nan
        return lbpdata
    else:
        print "The number of neighbouring points must be even and > 0!"



def get_lbp_patterns(number_of_bits, roinv, uniform):
    
    patterns = 0
    if roinv == 1 and uniform == 1: 
        patterns = np.zeros((number_of_bits + 2)) # zero pattern and non-uniform patterns included 
        # Set uniform an zero pattern
        patterns[0] = -1
        patterns[1] = 0
        for i in range(1, number_of_bits+1):
            val = 0
            for j in range(0, i):
                val += 2**j
            patterns[i+1] = val
    #elif roinv == 1 and uniform != 1:
    #elif roinv != 1 and uniform == 1:  
    #elif roinv != 1 and uniform != 1:      
    return patterns
                




# Will return a 3D-numpy array indicating the lbp-code neighbourhood frequencies in each pixel 
#
# INPUTS:
# 'data' numpy ndarray of lbp-codes, i.e. lbp-image 
# 'winSize' positive odd integer indicating the window size, e.g. 3, 5, 7, etc.
#
# OUTPUT:
# 'freqs' 3D-numpy array indicating the lbp-code frequencies in each pixel's neighbourhood

def get_freqs(data, winSize, neighs, roinv, uniform):
    
    cdef int rows, cols, col, row, y, x
    vals =  np.zeros((winSize**2,))
    rows = data.shape[0]
    cols = data.shape[1]   
    #uni = np.unique(data)
    #uni = uni[~np.isnan(uni)]
    uni = get_lbp_patterns(neighs, roinv, uniform)
    freqs = np.zeros((rows, cols, len(uni)))    
    dist = int(math.floor(float(winSize)/2.0)) # Distance to the edge of window from center pixel
    #neigh = range(-dist, dist+1) # Neighbourhood range in X- and Y-direction 
    # Loop whole image 
    for row in range(0, rows):
        for col in range(0, cols):
            index = 0
            number_of_elements = 0
            # Loop the neighbourhood of pixel [row, col]
            for y in range(-dist, dist+1):
                for x in range(-dist, dist+1): 
                    indY = row + y
                    indX = col + x
                    # Check that we are inside the image / matrix
                    if indY >= 0 and indY <= rows-1 and indX >= 0 and indX <= cols-1 and not np.isnan(data[indY, indX]):
                        vals[index] = data[indY, indX]
                        number_of_elements += 1
                    else: 
                        vals[index] = np.nan
                    index += 1
            index = 0
            # Calculate the frequencies in the neighbourhood
            for val in uni:
                if not np.isnan(val) and number_of_elements != 0:
                    freqs[row, col, index] = float(len(np.where(vals == val)[0])) / float(number_of_elements)
                else: # In this case all the elements are NaN-values
                    freqs[row, col, index] = 0
                index += 1
    return freqs, len(uni)





