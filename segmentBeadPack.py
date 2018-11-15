import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim
from matplotlib import cm

data = np.loadtxt("binaryMonoImage.txt",delimiter=',')
beadPack = np.reshape(data,(101,101,101))

dt = spim.distance_transform_edt(input=beadPack)
dt = spim.gaussian_filter(input=dt, sigma=0.4)
peaks = ps.network_extraction.find_peaks(dt=dt, r=4)

peaks = ps.network_extraction.trim_saddle_points(peaks=peaks, dt=dt)
peaks = ps.network_extraction.trim_nearby_peaks(peaks=peaks, dt=dt)

regions = ps.network_extraction.partition_pore_space(im=dt, peaks=peaks)

plt.figure(1)
plt.imshow((regions*beadPack)[:, :, 50], cmap=plt.cm.nipy_spectral)
plt.axis('off')

#regions.shape()
np.savetxt('beadPack_regions.txt', regions.flatten())

#print(regions)

    
#regions2 = ps.network_extraction.snow(dt)

#plt.figure(2)
#plt.imshow((regions2.peaks*beadPack)[:, :, 50], cmap=plt.cm.nipy_spectral)
#plt.axis('off')

#plt.show()
