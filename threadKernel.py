from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

lowerKernel = np.array(ImageOps.grayscale(Image.open("img/thread-kernel.png"))).astype(float) / 255
upperKernel = np.array(ImageOps.grayscale(Image.open("img/thread-kernel.png")).rotate(180)).astype(float) / 255
img = np.array(ImageOps.grayscale(Image.open("img/screw-thresh.png"))).astype(float) / 255

lowerKernel = (lowerKernel - 0.5) * 2
upperKernel = (upperKernel - 0.5) * 2

lowerConv = ndimage.convolve(img, lowerKernel)
upperConv = ndimage.convolve(img, upperKernel)

lowerConv = lowerConv / np.max(lowerConv)
upperConv = upperConv / np.max(upperConv)

threshold = 0.9
lowerThresh = lowerConv > threshold
upperThresh = upperConv > threshold
lowerConv[np.invert(lowerThresh)] = 0
upperConv[np.invert(upperThresh)] = 0

conv = lowerConv + upperConv

subplotRows = 1
subplotCols = 2

plt.subplot(subplotRows, subplotCols, 1)
plt.imshow(conv)

plt.subplot(subplotRows, subplotCols, 2)
plt.imshow(img)

plt.show()
