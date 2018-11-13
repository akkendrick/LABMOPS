import porespy as ps
import numpy as np
import matplotlib.pyplot as plt

im = ps.generators.blobs(shape=[200, 200], porosity=0.5, blobiness=2)
#plt.imshow(im)

mip = ps.filters.porosimetry(im)
data = ps.metrics.pore_size_distribution(mip)

#plt.imshow(mip)
plt.plot(*data,'b.-')

#plt.show()

# Attempt to read in our data
#open('beadPack.dat')
#with open('beadPack.dat') as f:
#    read_data = f.read()
#f.closed

#read_data.shape
#print(read_data)
data = np.loadtxt("beadPack.dat")
print(data)

beadPack = np.reshape(data,(101,101,101))
plt.imshow(beadPack[:,:,1])
plt.show()

mip = ps.filters.porosimetry(beadPack)
data = ps.metrics.pore_size_distribution(mip)

plt.plot(*data,'b.-')
plt.show()

