from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
import pyransac3d as pyrsc

import pdb

def convoleAndRANSAC(img: np.ndarray, kernel: np.ndarray):
    # convolve the image with the tread kernel
    conv = ndimage.convolve(img, kernel)

    # normalize to [0,1] so it displays nicely
    conv = conv - np.min(conv)
    conv = conv / np.max(conv)

    # only keep convolution results above some threshold
    # TODO: make this adaptive? might not be necessary when we're just dealing
    #  with silhouettes, but it could be useful for working straight off of
    #  images of screws
    threshold = 0.9
    thresholded = conv > threshold
    conv[np.invert(thresholded)] = 0

    candidatePixelsList = []

    # TODO: is there a less dumb way to do this?
    for i in range(thresholded.shape[0]):
        for j in range(thresholded.shape[1]):
            if (thresholded[i,j]):
                candidatePixelsList.append([j,i])

    # make arrays out of our pixel lists
    candidatePixels = np.array(candidatePixelsList)

    # set up the shape to accomodate a z dimension in each entry
    pointsArrayShape = (candidatePixels.shape[0], 3)

    # set the z for all to 0, we're only doing this because the RANSAC library
    #  I picked is designed for 3D
    candidatePoints = np.empty(pointsArrayShape)
    candidatePoints[:, 0:2] = candidatePixels
    candidatePoints[:, 2] = 0

    # run RANSAC on the candidate points
    line = pyrsc.Line()
    slope, intercept, inlierIndices = line.fit(candidatePoints, 0.01)

    # collect the inlier points
    inliers = np.empty((len(inlierIndices), 2))
    for i in range(len(inlierIndices)):
        inliers[i] = candidatePixels[inlierIndices[i]]

    return conv, inliers, slope, intercept

def printUsage():
    print("Usage: " + sys.argv[0] + " kernel img1 img2 img3 ...")

if (len(sys.argv) < 3):
    printUsage()

kernelImgFile = sys.argv[1]
kernelImg = ImageOps.grayscale(Image.open(kernelImgFile))
kernelImgInvert = ImageOps.invert(kernelImg)

lowerKernel = np.array(kernelImg).astype(float) / 255
lowerInsideKernel = np.array(kernelImgInvert.rotate(180)).astype(float) / 255

upperKernel = np.array(kernelImg.rotate(180)).astype(float) / 255
upperInsideKernel = np.array(kernelImgInvert).astype(float) / 255

lowerKernel = (lowerKernel - 0.5) * 2
upperKernel = (upperKernel - 0.5) * 2
lowerInsideKernel = (lowerInsideKernel - 0.5) * 2
upperInsideKernel = (upperInsideKernel - 0.5) * 2

imgs = []

for i in range(len(sys.argv) - 2):
    file = sys.argv[i+2]
    imgs.append(np.array(ImageOps.grayscale(Image.open(file))).astype(float) / 255)

convs = []
lowerInliers = []
upperInliers = []
lowerInsideInliers = []
upperInsideInliers = []

for i in range(len(imgs)):
    # convolve the image with the tread kernel and use RANSAC to find a cluster
    #  of points that represents the threadform
    # we do this twice because one of the kernels is upside down in order to
    #  get the threads in the lower half of the image
    lowerConv, _lowerInliers, _, _ = convoleAndRANSAC(imgs[i], lowerKernel)
    print("lower")
    upperConv, _upperInliers, _, _ = convoleAndRANSAC(imgs[i], upperKernel)
    print("upper")
    lowerInsideConv, _lowerInsideInliers, _, _ = convoleAndRANSAC(imgs[i], lowerInsideKernel)
    print("lower inside")
    upperInsideConv, _upperInsideInliers, _, _ = convoleAndRANSAC(imgs[i], upperInsideKernel)
    print("upper inside")

    convs.append(lowerConv + upperConv + lowerInsideConv + upperInsideConv)
    lowerInliers.append(_lowerInliers)
    upperInliers.append(_upperInliers)
    lowerInsideInliers.append(_lowerInsideInliers)
    upperInsideInliers.append(_upperInsideInliers)

    print(str(i+1) + "/" + str(len(imgs)))

subplotRows = 3
subplotCols = len(imgs)

for i in range(len(imgs)):
    plt.subplot(subplotRows, subplotCols, i + 1)
    plt.imshow(imgs[i])
    plt.plot(upperInliers[i][:, 0], upperInliers[i][:, 1], linewidth = 0, marker = '.', color = 'green')
    plt.plot(lowerInliers[i][:, 0], lowerInliers[i][:, 1], linewidth = 0, marker = '.', color = 'red')
    plt.plot(upperInsideInliers[i][:, 0], upperInsideInliers[i][:, 1], linewidth = 0, marker = '.', color = 'magenta')
    plt.plot(lowerInsideInliers[i][:, 0], lowerInsideInliers[i][:, 1], linewidth = 0, marker = '.', color = 'yellow')
    
    plt.subplot(subplotRows, subplotCols, i + 1 + len(imgs))
    plt.imshow(convs[i])

    plt.subplot(subplotRows, subplotCols, i + 1 + len(imgs) * 2)
    plt.imshow(convs[i] + imgs[i])
    plt.plot(upperInliers[i][:, 0], upperInliers[i][:, 1], linewidth = 0, marker = '.', color = 'green')
    plt.plot(lowerInliers[i][:, 0], lowerInliers[i][:, 1], linewidth = 0, marker = '.', color = 'red')
    plt.plot(upperInsideInliers[i][:, 0], upperInsideInliers[i][:, 1], linewidth = 0, marker = '.', color = 'magenta')
    plt.plot(lowerInsideInliers[i][:, 0], lowerInsideInliers[i][:, 1], linewidth = 0, marker = '.', color = 'yellow')

plt.show()
