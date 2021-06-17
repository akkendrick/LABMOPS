#%%

import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage as ski
import tqdm
import seaborn as sns
import pandas as pd
from pathlib import Path
import pickle as pkl
from sklearn.neighbors import KernelDensity
from  scipy.stats import gaussian_kde


cwd = Path().resolve()

#%%

# Define overall variables used to analyze the data
resolution = 16.81E-6 # adding resolution in meters
lowFlowVelCutoff = 6.5 * 10 ** float(-6) #0.000207 # 5.13 * 10 ** float(-5) # 0.5 * 10 ** float(-5)
poreDiamThresh = 20
poreVolumeCutoff = 38000
simPressure = 0.00005 # This is currently hardcoded!!
imageSize = 512
experimentTime = 90 * 10 ** float(-6) # specify NMR time

filePath = cwd / 'subBeadPackPy512_justSpheres.npy'
primaryImage = np.load(filePath)

filePath = cwd / 'finalSimFile3D512.npy'
secondaryImage = np.load(filePath)
primaryImage[primaryImage == 255] = 1

primaryImage = np.transpose(primaryImage)
secondaryImage = np.transpose(secondaryImage)

filePath = cwd / 'primarySnowOut.npy'
primarySnow = np.load(filePath)
filePath = cwd / 'primaryPoreInfo.p'
#primaryInfo = np.load(filePath,allow_pickle=True)
poreInfoPrimary = pkl.load(open(filePath,'rb'))

filePath = cwd / 'secondarySnowOut.npy'
secondarySnow = np.load(filePath)
filePath = cwd / 'secondaryPoreInfo.p'
#secondaryInfo = np.load(filePath,allow_pickle=True)
poreInfoSecondary = pkl.load(open(filePath,"rb"))


#%%

filePath = cwd / 'velocityFiles' / 'velocityNormCodeSecondary_0.00005.mat'
velSecondaryMat = sio.loadmat(filePath)
velDataNormSecondary = velSecondaryMat['velNorm']

filePath = cwd / 'velocityFiles' / 'velocityNormCodePrimary_0.00005.mat'
velPrimaryMat = sio.loadmat(filePath)
velDataNormPrimary = velPrimaryMat['velNorm']


#%%

slice = 35

fig, (p1, p2) = plt.subplots(1, 2)

fig.suptitle('Secondary pore space and velocity map')
p1.imshow(velDataNormSecondary[:,:,slice])
p2.imshow(secondaryImage[:,:,slice])

fig.savefig('secondaryPoreImage.png', dpi=300, facecolor='w', edgecolor='w')

#%%

plt.imshow(ps.tools.randomize_colors(secondarySnow[:,:,slice]), origin='lower')
plt.savefig('secondaryRegions.png', dpi=300, facecolor='w', edgecolor='w')


#%%

slice = 35

fig, (p1, p2) = plt.subplots(1, 2)

fig.suptitle('Primary pore space and velocity map')
p1.imshow(velDataNormPrimary[:,:,slice])
p2.imshow(primaryImage[:,:,slice])

fig.savefig('primaryPoreImage.png', dpi=300, facecolor='w', edgecolor='w')

#%% md

### Load all the data

#%%

intraPoreMeanPoreVelocity = np.load('intraPore_meanVel.npy')
intraPore_metric_PoreVelocity = np.load('intraPore_metricVel.npy')
intraPoreVolumeVector = np.load('intraPoreVolumeVector.npy')
intraPoreVelocityDataframe = np.load('intraPoreVelocityDataframe.npy')
intraPoreVolumeDataframe = np.load('intraPoreVolumeDataframe.npy')

maskedSecondaryMeanPoreVelocity = np.load('maskedPore_meanVel.npy')
maskedSecondary_metric_PoreVelocity = np.load('maskedPore_metricVel.npy')
maskedSecondaryPoreVolumeVector = np.load('maskedPoreVolumeVector.npy')
maskedSecondaryPoreVolumeDataframe = np.load('maskedPoreVolumeDataframe.npy')
maskedSecondaryPoreVelocityDataframe = np.load('maskedPoreVelocityDataframe.npy')

df_secondaryVelocity = pd.read_pickle('secondaryVelocities.pkl')

#%% md

## Format Dataframe


#%%

copyInterGrainPoreVol = np.array(df_secondaryVelocity["Intergrain Pore Volume"])
filtInterGrainVel = np.zeros(copyInterGrainPoreVol.shape)
for a in range(len(copyInterGrainPoreVol)):
    if np.isnan(copyInterGrainPoreVol[a]):
        filtInterGrainVel[a] = 0
    else:
        filtInterGrainVel[a] = copyInterGrainPoreVol[a]

#%%

df_secondaryVelocity["Intragrain Pore Volume"] = df_secondaryVelocity["Intragrain Pore Volume"] - filtInterGrainVel

#%%

df_secondaryVelocity

#%% md

## Plot Pore Volume

#%%

volumeBins = np.linspace(0, 200000, num=40)

fig, ax = plt.subplots()
sns.set_style('whitegrid')

sns.histplot(data=df_secondaryVelocity["Intergrain Pore Volume"],
     bins=volumeBins,color="orange", label="Intergranular Porosity")
sns.histplot(data=df_secondaryVelocity["Intragrain Pore Volume"],
     bins=volumeBins,color="dodgerblue", label="Intragranular Porosity")


#sns.displot(data=[df_secondaryVelocity["Intragrain Pore Volume"], df_secondaryVelocity["Intergrain Pore Volume"]],
#            bins=volumeBins)
plt.ylim([0,40])
plt.legend()
plt.ylabel('Pore Volume')

plt.plot([poreVolumeCutoff, poreVolumeCutoff],[0,200],'g',lw=1)

figStr = 'interAndIntraPoreVolume_pressure_'+str(simPressure)+'.png'

plt.savefig(figStr, dpi=300, facecolor='w', edgecolor='w')


#%% md

## Plot inter and intragranular porosity


#%%

bins = np.linspace(0.000002, 0.00005, num=20)
#np.append(bins,0.0001)
bins = np.append(bins, 1000)
bins = np.insert(bins, 0, 0)
bins = np.insert(bins, 1, 0.00000001)
bins = np.insert(bins, 2, 0.0000001)
bins = np.insert(bins, 3, 0.000001)


binnedIntraVel = pd.cut(df_secondaryVelocity["Intragrain Median Velocity"], bins=bins, right=False)
binnedInterVel = pd.cut(df_secondaryVelocity["Intergrain Median Velocity"], bins=bins, right=False)
df_secondaryVelocity["binned_intraVelocity"] = binnedIntraVel
df_secondaryVelocity["binned_interVelocity"] = binnedIntraVel

yMax = 3000
velBins = np.linspace(0, 0.000025, num=40)

fig, ax = plt.subplots()
#fig.suptitle('Intra and inter pore velocities', fontsize=20)
sns.histplot(data=df_secondaryVelocity["Intragrain Median Velocity"],bins=velBins,
             ax=ax,color="dodgerblue", label="Intraparticle Median Velocity")
sns.histplot(data=df_secondaryVelocity["Intergrain Median Velocity"], bins=velBins,
             ax=ax, color="orange", label="Intergrain Median Velocity")
plt.legend()
ax.set_ylim([0,200])
ax.set_ylabel('Median Velocity')

lowFlowVelCutoff = 6.5 * 10 ** float(-6) #0.000207 # 5.13 * 10 ** float(-5) # 0.5 * 10 ** float(-5)
ax.plot([lowFlowVelCutoff , lowFlowVelCutoff],[0,200],'g',lw=1)

figStr = 'interAndIntraPoreVelocity_pressure_'+str(simPressure)+'.png'

fig.savefig(figStr, dpi=300, facecolor='w', edgecolor='w')



#%% md

## Plot both pore volume and median velocity

#%%

########################################

yMax = 0.0001
xMax = 100000#np.max(df_secondaryVelocity["Intergrain Pore Volume"])

fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle('Pore volume vs Median pore velocity for Pore Pressure ='+str(simPressure), fontsize=20)
sns.scatterplot(data=df_secondaryVelocity,x="Intergrain Pore Volume", y='Intergrain Median Velocity',ax=axes[0])
axes[0].set_xlabel('Pore Volume (lattice units)', fontsize=18)
axes[0].set_ylabel('Median Pore Velocity', fontsize=18)
axes[0].set_title('Intergranular Pores', fontsize=20)
axes[0].set_ylim([0,yMax])
axes[0].set_xlim([0,xMax])
axes[0].plot([poreVolumeCutoff, poreVolumeCutoff],[0,yMax],'r',lw=5)
axes[0].plot([0,np.max(secondarySkeletonPoreVolume)],[lowFlowVelCutoff, lowFlowVelCutoff],'g',lw=5)

sns.scatterplot(data=df_secondaryVelocity, x="Intragrain Pore Volume", y='Intragrain Median Velocity',ax=axes[1])
axes[1].set_xlabel('Pore Volume (lattice units)', fontsize=18)
axes[1].set_ylabel('Median Pore Velocity', fontsize=18)
axes[1].set_title('Intragranular Pores', fontsize=20)
axes[1].set_ylim([0,yMax])
axes[1].set_xlim([0,xMax])
axes[1].plot([poreVolumeCutoff, poreVolumeCutoff],[0,yMax],'r',lw=5)
axes[1].plot([0,np.max(secondarySkeletonPoreVolume)],[lowFlowVelCutoff, lowFlowVelCutoff],'g',lw=5)

figStr = 'poreVolumeVsPoreVelocity_zoomIn'+'.png'

fig.savefig(figStr, dpi=300, facecolor='w', edgecolor='w')


#%%

df_secondaryVelocity.count()

#%% md

# Add diffusion broadening

#%%

D_0 = 2.023 * 10 ** float(-9) # for water at twenty degrees Celsius
T = 90 * 10 ** float(-6) # 90 ms, same as NMR experiment
diffusionLength = np.sqrt(D_0 * T)
diffusionVel = diffusionLength/T

randList = [-1,1]
diffusionSign = np.random.choice(randList,len(df_secondaryVelocity["All secondary regions"]))
diffusionDist = np.random.rand(len(df_secondaryVelocity["All secondary regions"]))

# FIXME: how is this defined really?
# figure out sigma of diffusion distribution
diffusionDist = np.random.normal(0,diffusionVel,len(df_secondaryVelocity["All secondary regions"]))
#diffusionDistScale = (diffusionDist-np.mean(diffusionDist)) / np.std(diffusionDist)
diffusionDistScale = (diffusionDist-np.min(diffusionDist)) / (np.max(diffusionDist)-np.min(diffusionDist))
diffusionDistScale = diffusionDistScale - np.mean(diffusionDistScale)
#diffusionAdd = diffusionAdd / np.mean(diffusionAdd)

#%%

fig, ax = plt.subplots()

sns.histplot(diffusionDistScale,kde=True)

figStr = 'diffDist'+'.png'

fig.savefig(figStr, dpi=300, facecolor='w', edgecolor='w')

#%%

secondaryMedianVel = df_secondaryVelocity["Median secondary pore velocity"]
secondaryMedianVel = np.array(secondaryMedianVel)
#secondaryVelScale = (secondaryMedianVel - np.nanmean(secondaryMedianVel)) / np.nanstd(secondaryMedianVel)
secondaryVelScale = (secondaryMedianVel - np.nanmin(secondaryMedianVel)) / (np.nanmax(secondaryMedianVel) - np.nanmin(secondaryMedianVel))

#%%

fig, ax = plt.subplots()

sns.histplot(secondaryVelScale)

figStr = 'simVelDist'+'.png'

fig.savefig(figStr, dpi=300, facecolor='w', edgecolor='w')

#%% md

### Plot all pore velocities

#%%

# filter by pore space
allVelocities = velDataNormSecondary[secondaryImage == True]
# smush into one dimension
allVelocities = np.squeeze(allVelocities)
# normalize by mean velocity
allVelocitiesNorm = allVelocities / np.mean(allVelocities)

#%%

allDisplacementNorm = allVelocitiesNorm * resolution * experimentTime # this is now in units of length

#%%

bins = np.linspace(0.000002, 0.00005, num=20)
#np.append(bins,0.0001)
bins = np.append(bins, 1000)
bins = np.insert(bins, 0, 0)
bins = np.insert(bins, 1, 0.00000001)
bins = np.insert(bins, 2, 0.0000001)
bins = np.insert(bins, 3, 0.000001)

binnedAllDisplacements = pd.cut(allDisplacementNorm, bins=bins, right=False)


#%%

sns.histplot(allVelocitiesNorm)

#%%

fig, ax = plt.subplots()

sns.histplot(allDisplacementNorm)

figStr = 'simulatedDisplacement'+'.png'

fig.savefig(figStr, dpi=300, facecolor='w', edgecolor='w')

#%% md

## Load NMR data

#%%

filePath = cwd / 'summedNMRDisp.mat'
NMR_dispData = sio.loadmat(filePath)
NMRDisp = NMR_dispData['summedDispData']

#%%

NMRDispScale = (NMRDisp - np.nanmean(NMRDisp)) / (np.nanmax(NMRDisp)-np.nanmin(NMRDisp))
NMRDispScale = np.transpose(NMRDispScale)
NMRDispScale = np.squeeze(NMRDispScale)

#%%

indices = [i for i in range(NMRDispScale.size)]

#%%

fig, ax = plt.subplots()

plt.plot(indices,NMRDispScale,axes=ax)

figStr = 'NMRDispHist'+'.png'

fig.savefig(figStr, dpi=300, facecolor='w', edgecolor='w')




#%%

 np.where(secondaryVelPlusDiffusion[3] == np.nan)

#%%

indices = [i for i in range(secondaryVelScale.size)]
secondaryVelScale = np.sort(secondaryVelScale)
secondaryVelScale = np.flip(secondaryVelScale)
plt.plot(indices,secondaryVelScale)


#%%

secondaryVelPlusDiffusion = secondaryVelScale + diffusionDistScale

fig, ax = plt.subplots()

sns.histplot(secondaryVelPlusDiffusion,ax=ax,kde=True)

figStr = 'simDispHist'+'.png'

fig.savefig(figStr, dpi=300, facecolor='w', edgecolor='w')

#%%

indices = np.where(np.isnan(secondaryVelPlusDiffusion), False, True)
secondaryVelPlusDiffusion = secondaryVelPlusDiffusion[indices]

#%%

indices = [i for i in range(secondaryVelPlusDiffusion.size)]
plt.plot(indices,secondaryVelPlusDiffusion)

#%%

testKDE = gaussian_kde(secondaryVelPlusDiffusion)
indices = np.random.rand(len(secondaryVelPlusDiffusion))
plotKDE = testKDE.evaluate(indices)

#%%

plt.plot(indices,plotKDE)


#%%

np.random.rand(10)

#%%

filePath = cwd / 'summedNMRDisp.mat'
NMR_dispData = sio.loadmat(filePath)
NMRDisp = NMR_dispData['summedDispData']

#%%

NMRDispScale = (NMRDisp - np.nanmean(NMRDisp)) / (np.nanmax(NMRDisp)-np.nanmin(NMRDisp))
NMRDispScale = np.transpose(NMRDispScale)
NMRDispScale = np.squeeze(NMRDispScale)

#%%

indices = [i for i in range(NMRDispScale.size)]

#%%

fig, ax = plt.subplots()

plt.plot(indices,NMRDispScale,axes=ax)

figStr = 'NMRDispHist'+'.png'

fig.savefig(figStr, dpi=300, facecolor='w', edgecolor='w')



#%% md

# Calculate flowing fraction for this data

#%%

intergrainPoreCount = df_secondaryVelocity["Intergrain Pore Volume"].count()
print('Number of inter particle pores is',str(intergrainPoreCount))

# Final porosity calculation
porosityCalc = ps.metrics.porosity(secondaryImage)
print('Total porosity:')
print(np.round(porosityCalc,2))

# Get grains
flippedImage = np.copy(secondaryImage)
flippedImage[secondaryImage == 1] = 0
flippedImage[secondaryImage == 0] = 1

#%% md

### Refer to other code for less/more mobile calculation
## lowFlowCompute.py
