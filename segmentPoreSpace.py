import porespy as ps
import matplotlib.pyplot as plt
import openpnm as op
import imageio
import scipy.ndimage as spim
from matplotlib import cm
import openpnm as op

im = ps.generators.cylinders(shape=[300, 300, 300], radius=5, nfibers=200)
plt.imshow(ps.visualization.sem(im), cmap=plt.cm.bone)
plt.axis('off')

dt = spim.distance_transform_edt(input=im)
dt = spim.gaussian_filter(input=dt, sigma=0.4)
peaks = ps.network_extraction.find_peaks(dt=dt, r=4)

peaks = ps.network_extraction.trim_saddle_points(peaks=peaks, dt=dt)
peaks = ps.network_extraction.trim_nearby_peaks(peaks=peaks, dt=dt)

regions = ps.network_extraction.partition_pore_space(im=dt, peaks=peaks)

plt.imshow((regions*im)[:, :, 100], cmap=plt.cm.nipy_spectral)
plt.axis('off')

net = ps.network_extraction.extract_pore_network(im=regions*im, dt=dt)

pn = op.network.GenericNetwork()
pn.update(net)
op.export_data(network=pn, filename='extracted_network', fileformat='VTK')

im = ps.network_extraction.align_image_with_openpnm(im)
imageio.mimsave('extracted_network.tif', sp.array(im, dtype=int))
