from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
import pyransac3d as pyrsc

import pdb

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
lowerInliers = []
upperInliers = []

for i in range(len(imgs)):
    # convolve the image with the tread kernel
    # we do this twice because one of the kernels is upside down in order to
    #  get the threads in the lower half of the image
    lowerConv = ndimage.convolve(imgs[i], lowerKernel)
    upperConv = ndimage.convolve(imgs[i], upperKernel)

    # normalize to [0,1] so it displays nicely
    lowerConv = lowerConv / np.max(lowerConv)
    upperConv = upperConv / np.max(upperConv)

    # only keep convolution results above some threshold
    # TODO: make this adaptive? might not be necessary when we're just dealing
    #  with silhouettes, but it could be useful for working straight off of
    #  images of screws
    threshold = 0.9
    lowerThresholded = lowerConv > threshold
    upperThresholded = upperConv > threshold
    lowerConv[np.invert(lowerThresholded)] = 0
    upperConv[np.invert(upperThresholded)] = 0

    convs.append(lowerConv + upperConv)

    lowerCandidatePixelsList = []
    upperCandidatePixelsList = []

    # TODO: is there a less dumb way to do this?
    for i in range(lowerThresholded.shape[0]):
        for j in range(lowerThresholded.shape[1]):
            if (lowerThresholded[i,j]):
                lowerCandidatePixelsList.append([j,i])
            if (upperThresholded[i,j]):
                upperCandidatePixelsList.append([j,i])

    # make arrays out of our pixel lists
    lowerCandidatePixels = np.array(lowerCandidatePixelsList)
    upperCandidatePixels = np.array(upperCandidatePixelsList)

    # set up the shape to accomodate a z dimension in each entry
    lowerPointsArrayShape = (lowerCandidatePixels.shape[0], 3)
    upperPointsArrayShape = (upperCandidatePixels.shape[0], 3)

    # set the z for all to 0, we're only doing this because the RANSAC library
    #  I picked is designed for 3D
    lowerCandidatePoints = np.empty(lowerPointsArrayShape)
    upperCandidatePoints = np.empty(upperPointsArrayShape)
    lowerCandidatePoints[:, 0:2] = lowerCandidatePixels
    upperCandidatePoints[:, 0:2] = upperCandidatePixels
    lowerCandidatePoints[:, 2] = 0
    upperCandidatePoints[:, 2] = 0

    # run RANSAC on the candidate points
    lowerLine = pyrsc.Line()
    lowerSlope, lowerIntercept, lowerInlierIndices = lowerLine.fit(lowerCandidatePoints, 0.01)
    upperLine = pyrsc.Line()
    upperSlope, upperIntercept, upperInlierIndices = upperLine.fit(upperCandidatePoints, 0.01)

    # collect the inlier points
    _lowerInliers = np.empty((len(lowerInlierIndices), 2))
    _upperInliers = np.empty((len(upperInlierIndices), 2))
    for i in range(len(lowerInlierIndices)):
        _lowerInliers[i] = lowerCandidatePixels[lowerInlierIndices[i]]
    for i in range(len(upperInlierIndices)):
        _upperInliers[i] = upperCandidatePixels[upperInlierIndices[i]]

    lowerInliers.append(_lowerInliers)
    upperInliers.append(_upperInliers)

subplotRows = 3
subplotCols = len(imgs)

for i in range(len(imgs)):
    plt.subplot(subplotRows, subplotCols, i + 1)
    plt.imshow(imgs[i])
    plt.plot(upperInliers[i][:, 0], upperInliers[i][:, 1], linewidth = 0, marker = '.', color = 'green')
    plt.plot(lowerInliers[i][:, 0], lowerInliers[i][:, 1], linewidth = 0, marker = '.', color = 'red')
    
    plt.subplot(subplotRows, subplotCols, i + 1 + len(imgs))
    plt.imshow(convs[i])

    plt.subplot(subplotRows, subplotCols, i + 1 + len(imgs) * 2)
    plt.imshow(convs[i] + imgs[i])
    plt.plot(upperInliers[i][:, 0], upperInliers[i][:, 1], linewidth = 0, marker = '.', color = 'green')
    plt.plot(lowerInliers[i][:, 0], lowerInliers[i][:, 1], linewidth = 0, marker = '.', color = 'red')

plt.show()
