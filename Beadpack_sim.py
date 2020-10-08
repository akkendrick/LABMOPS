import porespy as ps
import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as spim
import scipy.io as sio
import openpnm as op
import imageio
import numpy as np
import rawpy
import skimage.io
from PIL import Image, ImageEnhance
from porespy.filters import find_peaks, trim_saddle_points, trim_nearby_peaks
from porespy.tools import randomize_colors
from skimage.morphology import watershed
from skimage import exposure
from matplotlib import cm
from scipy import misc 
from skimage import data, io, filters
ps.visualization.set_mpl_style()

# data = np.fromfile("AW_DryScan_segm_650x650x294_8bitunsigned.raw")
# #data = np.loadtxt("AW_DryScan_segm_650x650x294_8bitunsigned.raw")

# #beadPack = np.ndarray.reshape(data,(650,650,254)) 
# plt.imshow(beadPack[:,:,80])
image = 'AW_DryScan_segm_650x650x294_8bitunsigned.tif'


# raw = rawpy.imread(path)
# #im = np.array(Image.open(path))

# rgb = raw.postprocess()


# imageio.imsave('default.tiff', rgb)

# scene_infile = open(path,'rb')
# scene_image_array = np.fromfile(scene_infile)

# beadPack = np.reshape(scene_image_array,(650,650,294)) 

# plt.imshow(beadPack[:,:,80],cmap=plt.cm.gray)
# plt.show()

# test = (beadPack[:,:,80])
# img = Image.fromarray(beadPack[:,:,80])
# img.save('testout.tiff')

# This only does a 2D slice, no good
# im = Image.open(image) 
# factor = 100000

# enhancer = ImageEnhance.Contrast(im)
# im_output = enhancer.enhance(factor)

# imarray = np.array(im_output) 
# imarray.shape 

# im_output.show()

# im = np.memmap('AW_DryScan_segm_650x650x294_8bitunsigned.raw', dtype=np.ubyte, shape=(650,650,294))

# test = im[:,:,30]
# plt.imshow(test, cmap = plt.cm.gray)
# plt.show()

imname = 'RandomSpherePackingMicrostructure.raw'

# im = skimage.io.imread(imname).astype(np.ubyte)

data = np.memmap(imname, dtype=np=uint8)
data.shape

dataShape = data.reshape((565,525,1071))
dataslice = dataShape[:,:,300]

edges = filters.sobel(dataslice)

data_eq = exposure.equalize_hist(dataslice)

image = dataslice;
plt.imshow(image,cmap='gray')
plt.show()




ps.io.to_vtk(data,path='out.vtk',divide=False,downsample=False, voxel_size=1, vox=False)

# from mayavi import mlab
# mlab.contour3d(dataShape)
# mlab.outline()