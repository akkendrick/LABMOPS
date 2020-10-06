import porespy as ps
import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as spim
import scipy.io as sio
import openpnm as op
import imageio
import numpy as np
import rawpy
import imageio
from PIL import Image
from porespy.filters import find_peaks, trim_saddle_points, trim_nearby_peaks
from porespy.tools import randomize_colors
from skimage.morphology import watershed
from matplotlib import cm

ps.visualization.set_mpl_style()

# data = np.fromfile("AW_DryScan_segm_650x650x294_8bitunsigned.raw")
# #data = np.loadtxt("AW_DryScan_segm_650x650x294_8bitunsigned.raw")

# #beadPack = np.ndarray.reshape(data,(650,650,254)) 
# plt.imshow(beadPack[:,:,80])
path = 'AW_DryScan_segm_650x650x294_8bitunsigned.raw'


# raw = rawpy.imread(path)
# #im = np.array(Image.open(path))

# rgb = raw.postprocess()


# imageio.imsave('default.tiff', rgb)

scene_infile = open(path,'rb')
scene_image_array = np.fromfile(scene_infile,dtype=np.uint8,count=-1)

beadPack = np.reshape(scene_image_array,(650,650,294)) 

plt.imshow(beadPack[:,:,80],cmap=plt.cm.gray)
test = (beadPack[:,:,80])


img = Image.fromarray(beadPack[:,:,80])
img.save('my.tiff')