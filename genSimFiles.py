# %%
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt

final_image = np.load('subBeadPackPy.npy')

# Pull out a sub image
orig_image = final_image
final_image = final_image[0:250, 0:250, 250]
im = final_image
copyImage = np.array(im)
copyImage.astype(bool)

# adding loop to simulate different images with different secondary porosities
radius=5
startPorosity=0.04
porosityStep = 0.01

for x in range(20):
    print(x)
    iterPorosity = startPorosity + x * porosityStep
    iterPorosity = round(iterPorosity,2)
    print(iterPorosity)

    copyImage[im == 0] = 1
    copyImage[im == 1] = 0

    im2 = ps.generators.overlapping_spheres(shape=[250, 250], radius=5, porosity=iterPorosity)
    imStep = copyImage.astype(bool) * im2
    fileName = "poreStructure_porosity_" + str(iterPorosity) + "_radius_" + str(radius)

    imStepCopy = np.array(copyImage)
    imStep[imStepCopy == 0] = 1
    #imStep[imStepCopy == 1] = 0

    poreSim = ps.filters.porosimetry(imStep)
    print(ps.metrics.porosity(imStep))
    sizeDist = ps.metrics.pore_size_distribution(poreSim)

    plt.plot(sizeDist.pdf)
    # plt.imshow(imStep)
    plt.show()
    input("Press Enter to continue...")

    plt.imsave(fileName+".png",imStep)#
    ps.io.to_palabos(imStep,fileName+".dat",0)
