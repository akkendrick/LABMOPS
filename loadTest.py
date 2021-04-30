
import numpy as np
import os

dir_path = os.path.abspath('')

imageSize = 512


fname = '/scratch/alexkend/velocityFiles/velocityCodeSecondary_0.00005.dat'

f = open(fname, 'r') # 'r' = read
data = np.loadtxt(fname)
f.close()

resolution = 16.81E-6 # adding resolution in meters

goodData = np.reshape(data, [imageSize, imageSize, imageSize, 3])

print(goodData.shape)