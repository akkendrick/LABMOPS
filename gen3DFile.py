# code to make simulation files in 3D
# use this to generate files for palabos fluid simulation
# also generates python files for reading
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt

final_image = np.load('beadpack_out.npy')

# current data is from Jan 15 2021
# https://www.digitalrocksportal.org/projects/175/sample/181/

imSize = 512

# Pull out a sub image
orig_image = final_image
final_image = final_image[0:imSize, 0:imSize, 0:imSize]

np.save('subBeadPackPy'+str(imSize)+'_justSpheres.npy', final_image)

im = final_image
copyImage = np.array(im)
copyImage.astype(bool)

fileName2 = "poreStructure3D"+str(imSize)+"_justPrimary"
vtkName2 = "poreStructure3DVTK"+str(imSize)+"_justPrimary"

ps.io.to_palabos(final_image,fileName2+".dat",0)
ps.io.to_vtk(final_image,vtkName2)

plt.imshow(final_image[0:imSize,0:imSize,2])
plt.title('Original Sim Pack')
plt.show()


radius=5
# adding secondary porosity
startPorosity=0.02
porosityStep = 0.02
numSteps = 14

secondPorosity = startPorosity + numSteps * porosityStep
secondPorosity = round(secondPorosity,2)

copyImage[im == 0] = 1
copyImage[im == 255] = 0

plt.imshow(copyImage[0:imSize,0:imSize,2])
plt.title('Inverted Sim Pack')
plt.show()

im2 = ps.generators.overlapping_spheres(shape=im.shape, radius=10, porosity=secondPorosity)
imStep = copyImage.astype(bool) * im2

fileName = "poreStructure3D"+str(imSize)+"_secondPorosity_" + str(secondPorosity) + "_radius_" + str(radius)
vtkName = "poreStructure3DVTK"+str(imSize)+"_secondPorosity_" + str(secondPorosity) + "_radius_" + str(radius)

imStepCopy = np.array(copyImage)
imStep[imStepCopy == 0] = 1


plt.imshow(imStep[0:imSize,0:imSize,2])
plt.title('Output Sim Pack')
plt.show()

print(imStep.shape)

np.save('finalSimFile3D'+str(imSize)+'.npy', imStep)

ps.io.to_palabos(imStep,fileName+".dat",0)
ps.io.to_vtk(imStep,vtkName)

print('Done')