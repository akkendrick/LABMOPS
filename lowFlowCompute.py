import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage as ski
import tqdm


imageSize = 250

primaryImage = np.load('subBeadPackPy250_justSpheres.npy')
secondaryImage = np.load('finalSimFile3D250.npy')
primaryImage[primaryImage == 255] = 1

primaryImage = np.transpose(primaryImage)
secondaryImage = np.transpose(secondaryImage)

velSecondaryMat = sio.loadmat('velNormSecondary.mat')
velDataNormSecondary = velSecondaryMat['velNorm']

velPrimaryMat = sio.loadmat('velNormPrimary.mat')
velDataNormPrimary = velPrimaryMat['velNorm']

# Define overall variables used to analyze the data
resolution = 16.81E-6 # adding resolution in meters
lowFlowVelCutoff = 1.42 * 10 ** float(-5) # 0.5 * 10 ** float(-5)
poreDiamThresh = 20
poreVolumeThresh = 35000
simPressure = 0.00005

# Secondary
#################################################
snowFiltSecondary = ps.filters.snow_partitioning(im=secondaryImage,r_max=4,sigma=0.4, return_all=True)
poreInfoSecondary = ps.networks.regions_to_network(snowFiltSecondary.regions, dt=snowFiltSecondary.dt)

nRegions = np.unique(snowFiltSecondary.regions).size
secondaryPoreDiamVec = np.zeros(nRegions,)
secondaryPoreVolumeVec = np.zeros(nRegions,)

for a in range(0, poreInfoSecondary['pore.diameter'].size):
    secondaryPoreDiamVec[a] = poreInfoSecondary['pore.diameter'][a]
    secondaryPoreVolumeVec[a] = poreInfoSecondary['pore.volume'][a]

secondaryRegions = snowFiltSecondary.regions

# Skeleton for secondary image
cubeSize = len(secondaryImage)
visit = np.zeros(len(np.unique(secondaryRegions)))
secondaryPoreDiamImage = np.zeros(secondaryImage.shape)
allSecondaryRegions = np.unique(secondaryRegions)
tempImage = np.zeros(secondaryRegions.shape)
secondaryPoreDiamVectorSkeleton = np.zeros(len(allSecondaryRegions))
secondary_metric_PoreVelocity =  []
secondaryMeanPoreVelocity =  []
secondarySkeletonPoreDiam = []
secondarySkeletonPoreRegion = []
secondarySkeletonPoreVolume = []

secondarySkelImage = ski.morphology.skeletonize(secondaryImage)

# Save data on the skeleton
secondaryVelocitiesSkeleton = []
secondaryPoreDiamSkeleton = []
secondaryPoreRegionSkeleton = []

secondaryFiltSkelImage = np.where(secondarySkelImage,True,False)
secondaryVelocitiesSkeleton = velDataNormSecondary[secondaryFiltSkelImage]
secondaryPoreRegionSkeleton = secondaryRegions[secondaryFiltSkelImage]

for a in tqdm.tqdm(range(0,len(allSecondaryRegions)), 'Secondary Regions Loop'):
            regionLabel = a
            currentRegion = a
            if regionLabel != 0: # Don't want grains to be counted
                if visit[regionLabel] == 0:
                    visit[regionLabel] = 1
                    # Adjust indices for pore information, regionlabel = 0 is pore 1 in poreInfo
                    poreDiam = secondaryPoreDiamVec[regionLabel - 1]
                    secondaryPoreDiamImage[secondaryRegions == regionLabel] = poreDiam

                    skeletonPoreVel = secondaryVelocitiesSkeleton[secondaryPoreRegionSkeleton == currentRegion]
                    secondaryMeanPoreVelocity = np.append(secondaryMeanPoreVelocity, np.mean(skeletonPoreVel))
                    secondary_metric_PoreVelocity = np.append(secondary_metric_PoreVelocity, np.median(skeletonPoreVel))

                    # Adjust indices between regionProps and snow algorithm
                    secondarySkeletonPoreDiam = np.append(secondarySkeletonPoreDiam,
                                                          poreInfoSecondary['pore.diameter'][currentRegion - 1])
                    secondarySkeletonPoreVolume = np.append(secondarySkeletonPoreVolume,
                                                            poreInfoSecondary['pore.volume'][currentRegion - 1])

                    secondarySkeletonPoreRegion = np.append(secondarySkeletonPoreRegion, currentRegion)


# Primary
#################################################
snowFiltPrimary = ps.filters.snow_partitioning(im=primaryImage,r_max=4,sigma=0.4, return_all=True)
poreInfoPrimary = ps.networks.regions_to_network(snowFiltPrimary.regions, dt=snowFiltPrimary.dt)

nRegions = np.unique(snowFiltPrimary.regions).size
primaryPoreDiamVec = np.zeros(nRegions,)
primaryPoreVolumeVec = np.zeros(nRegions,)

for a in range(0, poreInfoPrimary['pore.diameter'].size):
    primaryPoreDiamVec[a] = poreInfoPrimary['pore.diameter'][a]
    primaryPoreVolumeVec[a] = poreInfoPrimary['pore.volume'][a]

primaryRegions = snowFiltPrimary.regions

# Skeleton for primary image
cubeSize = len(primaryImage)
visit = np.zeros(len(np.unique(primaryRegions)))
primaryPoreDiamImage = np.zeros(primaryImage.shape)
allPrimaryRegions = np.unique(primaryRegions)
tempImage = np.zeros(primaryRegions.shape)
primaryPoreDiamVectorSkeleton = np.zeros(len(allPrimaryRegions))
primary_metric_PoreVelocity =  []
primaryMeanPoreVelocity =  []
primarySkeletonPoreDiam = []
primarySkeletonPoreRegion = []
primarySkeletonPoreVolume = []
primarySkelImage = ski.morphology.skeletonize(primaryImage)

# Save data on the skeleton
primaryVelocitiesSkeleton = []
primaryPoreDiamSkeleton = []
primaryPoreRegionSkeleton = []

primaryFiltSkelImage = np.where(primarySkelImage,True,False)
primaryVelocitiesSkeleton = velDataNormPrimary[primaryFiltSkelImage]
primaryPoreRegionSkeleton = primaryRegions[primaryFiltSkelImage]

for a in tqdm.tqdm(range(0,len(allPrimaryRegions)), 'Primary Regions loop'):
            regionLabel = a
            currentRegion = a
            if regionLabel != 0: # Don't want grains to be counted
                if visit[regionLabel] == 0:
                    visit[regionLabel] = 1
                    # Adjust indices for pore information, regionlabel = 0 is pore 1 in poreInfo
                    poreDiam = primaryPoreDiamVec[regionLabel - 1]
                    primaryPoreDiamImage[primaryRegions == regionLabel] = poreDiam

                    skeletonPoreVel = primaryVelocitiesSkeleton[primaryPoreRegionSkeleton == currentRegion]
                    primaryMeanPoreVelocity = np.append(primaryMeanPoreVelocity, np.mean(skeletonPoreVel))
                    primary_metric_PoreVelocity = np.append(primary_metric_PoreVelocity, np.median(skeletonPoreVel))

                    # Adjust indices between regionProps and snow algorithm
                    primarySkeletonPoreDiam = np.append(primarySkeletonPoreDiam,
                                                        poreInfoPrimary['pore.diameter'][currentRegion - 1])
                    primarySkeletonPoreVolume = np.append(primarySkeletonPoreVolume,
                                                          poreInfoPrimary['pore.volume'][currentRegion - 1])

                    primarySkeletonPoreRegion = np.append(primarySkeletonPoreRegion, currentRegion)





# Plot data
########################################################################################################

yMax = 0.0001

fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle('Both Samples', fontsize=24)

axes[0].scatter(primarySkeletonPoreVolume, primary_metric_PoreVelocity)
axes[0].set_xlabel('Pore Volume (lattice units)', fontsize=18)
axes[0].set_ylabel('Median Pore Skeleton Velocity', fontsize=18)
axes[0].set_title('Primary Porosity', fontsize=20)
axes[0].set_ylim([0,yMax])
axes[0].set_xlim([0,np.max(primarySkeletonPoreVolume)])
axes[0].plot([poreVolumeThresh, poreVolumeThresh],[0,yMax],'r',lw=5)
axes[0].plot([0,np.max(primarySkeletonPoreVolume)],[lowFlowVelCutoff, lowFlowVelCutoff],'g',lw=5)

axes[1].scatter(secondarySkeletonPoreVolume, secondary_metric_PoreVelocity)
axes[1].set_xlabel('Pore Volume (lattice units)', fontsize=18)
axes[1].set_ylabel('Median Pore Skeleton Velocity', fontsize=18)
axes[1].set_title('Secondary Porosity', fontsize=20)
axes[1].set_ylim([0,yMax])
axes[1].set_xlim([0,np.max(secondarySkeletonPoreVolume)])
axes[1].plot([poreVolumeThresh, poreVolumeThresh],[0,yMax],'r',lw=5)
axes[1].plot([0,np.max(secondarySkeletonPoreVolume)],[lowFlowVelCutoff, lowFlowVelCutoff],'g',lw=5)

figStr = 'poreRegionPlot_pressure_'+str(simPressure)+'.png'
fig.show()
fig.savefig(figStr, dpi=300, facecolor='w', edgecolor='w')


# Output flow region info
########################################################################################################
bigSecondaryPores = secondarySkeletonPoreVolume[secondarySkeletonPoreVolume > poreVolumeThresh]
bigSecondaryPoreRegions = secondarySkeletonPoreRegion[secondarySkeletonPoreVolume > poreVolumeThresh]
bigSecondaryPoreVel = secondary_metric_PoreVelocity[secondarySkeletonPoreVolume > poreVolumeThresh]
#lowFlowPores = bigSecondaryPores[]

smallSecondaryPores = secondarySkeletonPoreVolume[secondarySkeletonPoreVolume <= poreVolumeThresh]
smallSecondaryPoreRegions = secondarySkeletonPoreRegion[secondarySkeletonPoreVolume <= poreVolumeThresh]
smallSecondaryPoreVel = secondary_metric_PoreVelocity[secondarySkeletonPoreVolume <= poreVolumeThresh]


lowFlowBigPoreRegions = bigSecondaryPoreRegions[bigSecondaryPoreVel <= lowFlowVelCutoff]
lowFlowBigPoreDiam = bigSecondaryPores[bigSecondaryPoreVel <= lowFlowVelCutoff]
lowFlowBigPoreFlow = bigSecondaryPoreVel[bigSecondaryPoreVel <= lowFlowVelCutoff]

highFlowBigPoreDiam = bigSecondaryPores[bigSecondaryPoreVel > lowFlowVelCutoff]

print('There are',str(len(lowFlowBigPoreDiam)),'low flow pores of large pore diameter')
print('There are',str(len(highFlowBigPoreDiam)),'high flow pores of large pore diameter')

lowFlowBigIMOut = np.zeros(secondaryImage.shape)
lowFlowBigIM = np.zeros(secondaryImage.shape)
lowFlowBigCount = 0

highFlowBigIMOut = np.zeros(secondaryImage.shape)
highFlowBigIM = np.zeros(secondaryImage.shape)
highFlowBigCount = 0

for a in tqdm.tqdm(range(0,len(secondarySkeletonPoreRegion)),'Flow Velocity Loop'):

    currentRegion = secondarySkeletonPoreRegion[a]

    if currentRegion in lowFlowBigPoreRegions:
        lowFlowBigIMOut[secondaryRegions == currentRegion] = 255
        lowFlowBigIM[secondaryRegions == currentRegion] = 1
        lowFlowBigCount = lowFlowBigCount + 1
    else:
        if secondarySkeletonPoreVolume[a] > poreVolumeThresh:
            highFlowBigIMOut[secondaryRegions == currentRegion] = 255
            highFlowBigIM[secondaryRegions == currentRegion] = 1
            highFlowBigCount = highFlowBigCount + 1
            #print('High Flow Big Pore', str(currentRegion))


print('Number of big pores with low flow is',str(lowFlowBigCount))
print('Number of big pores with high flow is',str(highFlowBigCount))

bigPoreIMOut = np.zeros(secondaryImage.shape)
bigPoreIM = np.zeros(secondaryImage.shape)
bigPoreCount = 0

smallPoreIMOut = np.zeros(secondaryImage.shape)
smallPoreIM = np.zeros(secondaryImage.shape)

for a in tqdm.tqdm(range(0,len(secondarySkeletonPoreRegion)),'Pore Volume Loop'):

    currentRegion = secondarySkeletonPoreRegion[a]
    currentPoreVolume = secondarySkeletonPoreVolume[a]

    if currentPoreVolume > poreVolumeThresh:
        bigPoreIMOut[secondaryRegions == currentRegion] = 255
        bigPoreIM[secondaryRegions == currentRegion] = 1
        bigPoreCount = bigPoreCount + 1
    else:
        smallPoreIMOut[secondaryRegions == currentRegion] = 125
        smallPoreIM[secondaryRegions == currentRegion] = 1

print('Number of big pores is',str(bigPoreCount))

# Final porosity calculation
porosityCalc = ps.metrics.porosity(secondaryImage)
print('Total porosity:')
print(np.round(porosityCalc,2))

# Get grains
flippedImage = np.copy(secondaryImage)
flippedImage[secondaryImage == 1] = 0
flippedImage[secondaryImage == 0] = 1

sumPorosity = np.sum(secondaryImage)/(np.sum(flippedImage)+np.sum(secondaryImage))
print('Summed porosity to check value')
print(np.round(sumPorosity,2))

lessMobileOnes = np.sum(lowFlowBigIM)
total = np.sum(flippedImage) + np.sum(secondaryImage)

lessMobilePorosity = lessMobileOnes/total
print('Less mobile porosity estimate:')
print(np.round(lessMobilePorosity,2))

print('Less mobile fraction of porosity')
lessMobileFrac = (lessMobilePorosity/porosityCalc)
print(np.round(lessMobileFrac,2))

bigPoreSum = np.sum(bigPoreIM)
smallPoreSum = np.sum(smallPoreIM)

allPoreSum = smallPoreSum + bigPoreSum

fracBigPores = bigPoreSum/(smallPoreSum+bigPoreSum)
print('Fraction of pore space identified as big pores is',str(np.round(fracBigPores,2)))

porosityEst = (bigPoreSum + smallPoreSum) / (np.sum(flippedImage)+(bigPoreSum+smallPoreSum))
#print('Sum porosity to check porosity value')
#print(porosityEst)

bigPoresLowFlow = np.sum(lowFlowBigIM)
bigPoresFlow = np.sum(highFlowBigIM)

#print(bigPoresLowFlow+bigPoresFlow)

fracBigPoresFlow = bigPoresFlow / (bigPoresLowFlow + bigPoresFlow)
fracBigPoresLowFlow = bigPoresLowFlow / (bigPoresLowFlow + bigPoresFlow)

print('Fraction of big pore space that flows is',str(np.round(fracBigPoresFlow,2)))
print('Fraction of big pore space with low flow is',str(np.round(fracBigPoresLowFlow,2)))
