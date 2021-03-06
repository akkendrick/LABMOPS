import porespy as ps
import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as spim
import scipy.io as sio
import pickle 
from matplotlib import cm

### Load pore structure and LB simulation data

velocityNormData = np.loadtxt('weibullCubeVelocityNorm_300.dat')
simSize = 300
velocityNormData = np.reshape(velocityNormData,(simSize,simSize,simSize))

data = np.loadtxt("weibull300_cube.txt",delimiter=',')
beadPack = np.reshape(data,(simSize,simSize,simSize))

beadPack = np.transpose(beadPack,(1,2,0))

plt.figure(2)
plt.imshow(velocityNormData[:,:,100], cmap=plt.cm.nipy_spectral)

plt.imshow(beadPack[:,:,100],cmap=plt.cm.nipy_spectral)

plt.imshow((beadPack*velocityNormData)[:,:,100],cmap=plt.cm.nipy_spectral)

### Compute pore network information

dt = spim.distance_transform_edt(input=beadPack)
dt = spim.gaussian_filter(input=dt, sigma=0.4)
peaks = ps.network_extraction.find_peaks(dt=dt, r=4)

peaks = ps.network_extraction.trim_saddle_points(peaks=peaks, dt=dt)
peaks = ps.network_extraction.trim_nearby_peaks(peaks=peaks, dt=dt)

regions = ps.network_extraction.partition_pore_space(im=dt, peaks=peaks)

plt.figure(1)
plt.imshow((regions*beadPack)[:, :, 50], cmap=plt.cm.nipy_spectral)
plt.axis('off')

im = regions*beadPack
im = im.astype(np.int64)
net = ps.network_extraction.extract_pore_network(im=im, dt=dt)

cubeSize = len(beadPack)

velocities = {}

regionMap = regions*beadPack

for a in range(0, cubeSize): 
    for b in range(0, cubeSize):
        for c in range(0, cubeSize):
            
            #print(beadPack[a,b,c])
                        
            key = str(regionMap[a,b,c])
            if key != '0.0':
                if key in velocities:
                    velocities[key].append(velocityNormData[a,b,c])
                else:
                    velocities.setdefault(key,[])
                    velocities[key].append(velocityNormData[a,b,c])
        
            
numPores = len(net['pore.label'])
avgVelocity = []
for a in range(0, numPores):
    key = "%0.1f" % net['pore.label'][a]

    avgVelocity.append(sum(velocities[key])/len(velocities[key]))           

net['avgVelocity'] = avgVelocity

#props = ps.metrics.regionprops_3D(im)


#plt.figure(5)
#plt.scatter(net['pore.diameter'],net['avgVelocity'])
#plt.show

# Save output data via pickle
outFile = 'weibullCubeNetworkInfo_300.pkl'

output = open(outFile,'wb')
pickle.dump(net, output)
pickle.dump(velocities, output)

#pickle.dump(props, output)

output.close()


