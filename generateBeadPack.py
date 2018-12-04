import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim
from matplotlib import cm

shape = [300,300,300]
radius = 15
porosity = 0.34

beadPack1 = ps.generators.overlapping_spheres(shape,radius,porosity)

plt.figure(1)
plt.imshow(beadPack1[:,:,50])

#beadPack2 = ps.generators.sphere_pack(shape,radius,offset=0,packing='sc')

#plt.figure(2)
#plt.imshow(beadPack2[:,:,50])






plt.show()
