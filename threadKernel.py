from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import sys

def printUsage():
    print("Usage: " + sys.argv[0] + " kernel img1 img2 img3 ...")

if (len(sys.argv) < 3):
    printUsage()


kernelImg = sys.argv[1]
lowerKernel = np.array(ImageOps.grayscale(Image.open(kernelImg))).astype(float) / 255
upperKernel = np.array(ImageOps.grayscale(Image.open(kernelImg)).rotate(180)).astype(float) / 255

lowerKernel = (lowerKernel - 0.5) * 2
upperKernel = (upperKernel - 0.5) * 2

imgs = []

for i in range(len(sys.argv) - 2):
    file = sys.argv[i+2]
    imgs.append(np.array(ImageOps.grayscale(Image.open(file))).astype(float) / 255)

convs = []

for i in range(len(imgs)):
    lowerConv = ndimage.convolve(imgs[i], lowerKernel)
    upperConv = ndimage.convolve(imgs[i], upperKernel)

    lowerConv = lowerConv / np.max(lowerConv)
    upperConv = upperConv / np.max(upperConv)

    threshold = 0.9
    lowerThresh = lowerConv > threshold
    upperThresh = upperConv > threshold
    lowerConv[np.invert(lowerThresh)] = 0
    upperConv[np.invert(upperThresh)] = 0

    convs.append(lowerConv + upperConv)

subplotRows = 3
subplotCols = len(imgs)

for i in range(len(imgs)):
    plt.subplot(subplotRows, subplotCols, i + 1)
    plt.imshow(imgs[i])
    
    plt.subplot(subplotRows, subplotCols, i + 1 + len(imgs))
    plt.imshow(convs[i])

    plt.subplot(subplotRows, subplotCols, i + 1 + len(imgs) * 2)
    plt.imshow(convs[i] + imgs[i])

plt.show()
