import numpy as np
import porespy as ps
import matplotlib.pyplot as plt

final_image = np.load('subBeadPackPy.npy')

# current data is from Jan 15 2021
# https://www.digitalrocksportal.org/projects/175/sample/181/

# Pull out a sub image
orig_image = final_image
final_image = final_image[0:250, 0:250, 0:250]
im = final_image
copyImage = np.array(im)
copyImage.astype(bool)

# adding secondary porosity
radius=5
startPorosity=0.04
porosityStep = 0.01
numSteps = 14

secondPorosity = startPorosity + numSteps * porosityStep
secondPorosity = round(secondPorosity,2)

copyImage[im == 0] = 1
copyImage[im == 1] = 0

im2 = ps.generators.overlapping_spheres(shape=im.shape, radius=10, porosity=secondPorosity)
imStep = copyImage.astype(bool) * im2
fileName = "poreStructure3D_porosity_" + str(secondPorosity) + "_radius_" + str(radius)
vtkName = "poreStructureVTK_" + str(secondPorosity) + "_radius_" + str(radius)

imStepCopy = np.array(copyImage)
imStep[imStepCopy == 0] = 1

plt.imshow(imStep[0:250,0:250,20])
plt.show()

print(imStep.shape)

ps.io.to_palabos(imStep,fileName+".dat",0)
ps.io.to_vtk(imStep,vtkName)
print('Done')