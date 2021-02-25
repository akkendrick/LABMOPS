import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
import openpnm as pnm

poreImage = np.load('subBeadPackPy.npy')
poreVelocity = np.load('poreStructure3D_velNorm.dat')

im_3d = ps.visualization.show_3D(im)
plt.imshow(im_3d, cmap=plt.cm.moagma);

snow = ps.networks.snow(im=im, boundary_faces=['right'])
proj = op.io.PoreSpy.import_data(snow)
print(proj)