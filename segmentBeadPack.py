import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim
import scipy.io as sio
from matplotlib import cm

data = np.loadtxt("polyImage_256.txt",delimiter=',')
beadPack = np.reshape(data,(256,256,256))

dt = spim.distance_transform_edt(input=beadPack)
dt = spim.gaussian_filter(input=dt, sigma=0.4)
peaks = ps.network_extraction.find_peaks(dt=dt, r=4)

peaks = ps.network_extraction.trim_saddle_points(peaks=peaks, dt=dt)
peaks = ps.network_extraction.trim_nearby_peaks(peaks=peaks, dt=dt)

regions = ps.network_extraction.partition_pore_space(im=dt, peaks=peaks)

plt.figure(1)
plt.imshow((regions*beadPack)[:, :, 50], cmap=plt.cm.nipy_spectral)
plt.axis('off')

plt.figure(3)
plt.imshow((peaks*beadPack)[:,:,50],cmap=plt.cm.nipy_spectral)

im = regions*beadPack
im = im.astype(np.int64)
net = ps.network_extraction.extract_pore_network(im=im, dt=dt)

props = ps.metrics.regionprops_3D(im)

# To see all info stored use
net.keys()

# Try saving dictionary 
sio.savemat('polyPoreNetwork_256',net)
sio.savemat('polyRegionProps_256',props)
#regions.shape()
#np.savetxt('beadPack_regions.txt', regions.flatten())

#print(regions)


# Pore space segmentation does not appear to be working!!! Pores are way too big.
# What's the best way to deal with this? Is the simulation resolution too low?
# Need more grid cells to properly identify pores?

# Ideas:
# Try running on model pore structures for verification
# Run SNOW with more iterations?
# Read paper for suggestions about this!! 

# regions2 = ps.network_extraction.snow(beadPack)

# plt.figure(2)

# plt.imshow((regions2.regions*regions2.im)[:, :, 50], cmap=plt.cm.nipy_spectral)
# plt.axis('off')

# poreFlags = regions2.regions * regions2.im
# np.savetxt('beadPack_regions.txt',poreFlags.flatten())


plt.show()
