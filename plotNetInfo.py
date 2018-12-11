import porespy as ps
import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as spim
import scipy.io as sio
import pickle 
from matplotlib import cm

fileName = 'weibullCubeNetworkInfo_300.pkl'
inputFile = open(fileName,'rb')

net = pickle.load(inputFile)
velocities = pickle.load(inputFile)

plt.figure(1)
plt.scatter(net['pore.surface_area'], net['avgVelocity'])
plt.axis([0, 12000, -1.0e-06, 1.3e-05])
plt.xlabel('Pore Surface Area')
plt.ylabel('Avg Pore Velocity')


plt.figure(2)
plt.scatter(net['pore.diameter'],net['avgVelocity'])
plt.axis([0,30,-1.0e-06, 1.3e-05])
plt.xlabel('Pore Diameter')
plt.ylabel('Avg Pore Velocity')

plt.figure(3)
plt.scatter(net['pore.volume'], net['avgVelocity'])
plt.axis([0,50000,-1.0e-06, 1.3e-05])
plt.xlabel('Pore Volume')
plt.ylabel('Avg Pore Velocity')


# Calculate surface to volume ratio
SoverV = net['pore.surface_area'] / net['pore.volume']

plt.figure(4)
plt.scatter(SoverV, net['avgVelocity'])
plt.axis([0,1.0,-1.0e-06, 1.2e-05])
plt.xlabel('Surface to volume ratio')
plt.ylabel('Avg Pore Velocity')

VoverS = net['pore.volume'] / net['pore.surface_area']

plt.figure(5)
plt.scatter(VoverS, net['avgVelocity'])
plt.axis([0.8,6.0,-1.0e-06,1.2e-05])
plt.xlabel('Volume to surface ratio')
plt.ylabel('Avg Pore Velocity')

# Try crudely estimating relaxation time? 
rho = 20 # Based on Luo et al 2015 JMR paper
T2 = VoverS/rho 

plt.figure(6)
plt.hist(T2, bins='auto')
plt.title('T2 Distribution?')
plt.xscale('log')


plt.show()
