import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
#from mayavi import mlab

#im = ps.generators.blobs(shape=[200, 200], porosity=0.5, blobiness=2)
#plt.imshow(im)

#mip = ps.filters.porosimetry(im)
#data = ps.metrics.pore_size_distribution(mip)

#plt.imshow(mip)
#plt.plot(*data,'b.-')

#plt.show()

# Attempt to read in our data
#open('beadPack.dat')
#with open('beadPack.dat') as f:
#    read_data = f.read()
#f.closed

#read_data.shape
#print(read_data)


# Load binary bead pack data
# 0's are solid, 1's are pore space

data = np.loadtxt("binaryMonoImage.txt",delimiter=',')
#print(data)

beadPack = np.reshape(data,(101,101,101))
plt.figure(1)
plt.imshow(beadPack[:,:,1])

mip = ps.filters.porosimetry(beadPack)
plt.figure(2)
plt.imshow(mip[:,:,1])

np.savetxt('poreSize.txt', mip.flatten())
np.savetxt('beadPack_py.txt', beadPack.flatten())

#mlab.volume_slice(10,10,10, mip)
#mlab.show()

# mip should now be an image indicating sphere radius at which it becomes
# accessbile from the "inlets"


data = ps.metrics.pore_size_distribution(mip)
plt.figure(3)
plt.plot(*data,'b.-')

plt.show()


