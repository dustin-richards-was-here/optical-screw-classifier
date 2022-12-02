from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import sys

def printUsage():
    print("Usage: " + sys.argv[0] + " img ylevel_for_fft")

if (len(sys.argv) < 3):
    printUsage()

file = sys.argv[1]
yLevel = int(sys.argv[2])

img = np.array(ImageOps.grayscale(Image.open(file))).astype(float) / 255

strip = img[yLevel, :]

plt.plot(strip)
plt.show()
