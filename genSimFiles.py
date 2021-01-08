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

    im2 = ps.generators.overlapping_spheres(shape=[250, 250], radius=10
                                            , porosity=iterPorosity)
    imStep = copyImage.astype(bool) * im2
    fileName = "poreStructure_porosity_" + str(iterPorosity) + "_radius_" + str(radius)

    imStepCopy = np.array(copyImage)
    imStep[imStepCopy == 0] = 1
    #imStep[imStepCopy == 1] = 0

    lt = ps.filters.local_thickness(imStep,sizes=400)
    #poreSim = ps.filters.porosimetry(imStep, sizes=10)
    print(ps.metrics.porosity(imStep))
    sizeDist = ps.metrics.pore_size_distribution(lt)

   # chordSim = ps.metrics.chord_counts(im)
    # chordSim = ps.filters.apply_chords(imStep)
    #sizeDist = ps.metrics.chord_length_distribution(chordSim)

    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax1.set_ylabel('PDF frequency?')
    ax1.set_xlabel('Pore Size?')
    ax1.set_title('PDF of pore size distribution')
    ax1.plot(sizeDist.pdf)

    ax2 = fig.add_subplot(122)
    ax2.set_title('Pore Image for \n secondary porosity of ' + str(iterPorosity))
    ax2.imshow(imStep)


    plt.savefig(fileName+"_plot.png")
    plt.show()

    input("Press Enter to continue...")

    ps.io.to_palabos(imStep,fileName+".dat",0)
