from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
import pyransac3d as pyrsc
import math

import pdb

# convolve img (should be a screw aligned along and centered with the y-axis)
#  and kernel (should be a thread point) and use RANSAC to fit a line to the
#  points of the threads matched by the kernel
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

    # the RANSAC library returns a 3D slope vector, convert it to rise/run
    slope = math.tan(math.atan2(slope[1], slope[0]))

    return conv, inliers.astype(int), slope, intercept[1]

# grabs a line of pixels from the image along some line defined by the given
#  slope, intercept, and intercept y-offset between the given x limits
def getLineOfPixels(img: np.ndarray, slope: float, intercept: float, yOffset: float, xLimits: tuple[int]):
    line = np.empty(xLimits[1] - xLimits[0])

    for x in range(xLimits[0], xLimits[1]):
        y = slope * x + intercept + yOffset
        line[x - xLimits[0]] = img[int(y)][x]

    return line

# starting from a given line, take the average intensity of pixels across the line.
#  repeat until line is at center of image. return list of intensities.
def getIntensityToCenter(img: np.ndarray, slope: float, intercept: float, xLimits: tuple[int]):
    yOffset = 0
    intensities = np.empty(int(abs(intercept - img.shape[0] / 2)))

    direction = 0
    if (intercept > img.shape[0] / 2):
        direction = -1
    else:
        direction = 1

    for i in range(len(intensities)):
        avgIntensity = np.average(getLineOfPixels(img, slope, intercept, yOffset, xLimits))
        intensities[i] = avgIntensity
        yOffset += direction

    return intensities

def printUsage():
    print("Usage: " + sys.argv[0] + " kernel img1 img2 img3 ...")

def main():
    if (len(sys.argv) < 3):
        printUsage()

    kernelImgFile = sys.argv[1]
    kernelImg = ImageOps.grayscale(Image.open(kernelImgFile))

    lowerKernel = np.array(kernelImg).astype(float) / 255
    upperKernel = np.array(kernelImg.rotate(180)).astype(float) / 255

    lowerKernel = (lowerKernel - 0.5) * 2
    upperKernel = (upperKernel - 0.5) * 2

    imgs = []

    for i in range(len(sys.argv) - 2):
        file = sys.argv[i+2]
        imgs.append(np.array(ImageOps.grayscale(Image.open(file))).astype(float) / 255)

    convs = []
    lowerInliers = []
    upperInliers = []
    lowerLineY = []
    upperLineY = []
    lowerIntensityToCenter = []
    upperIntensityToCenter = []
    lowerThreadFrequencySamples = []
    ffts = []

    for i in range(len(imgs)):
        # convolve the image with the tread kernel and use RANSAC to find a cluster
        #  of points that represents the threadform
        # we do this twice because one of the kernels is upside down in order to
        #  get the threads in the lower half of the image
        lowerConv, _lowerInliers, lowerSlope, lowerIntercept = convoleAndRANSAC(imgs[i], lowerKernel)
        upperConv, _upperInliers, upperSlope, upperIntercept = convoleAndRANSAC(imgs[i], upperKernel)

        # combine the two convolution images, really just for visualizing
        convs.append(lowerConv + upperConv)

        # save the array of inlier points
        lowerInliers.append(_lowerInliers)
        upperInliers.append(_upperInliers)

        # calculate the points of a line for either side of the screw based on the
        #  RANSAC results from above
        _lowerLineY = np.empty(imgs[i].shape[1])
        _upperLineY = np.empty(imgs[i].shape[1])

        # TODO: optimize with numpy
        for j in range(imgs[i].shape[1]):
            _lowerLineY[j] = lowerSlope * j + lowerIntercept
            _upperLineY[j] = upperSlope * j + upperIntercept

        lowerLineY.append(_lowerLineY)
        upperLineY.append(_upperLineY)

        # find the x coordinates of the furthest left and right inlier point
        lowerXLimArgmin = np.argmin(_lowerInliers[:, 0])
        lowerXLimArgmax = np.argmax(_lowerInliers[:, 0])
        lowerXLims = (_lowerInliers[lowerXLimArgmin, 0], _lowerInliers[lowerXLimArgmax, 0])

        # start with a line between these two extreme points. take the average
        #  intensity of the image across this line. step the line's y-intercept
        #  closer to the centerline of the image, take the average intensity.
        #  repeat until at the center.
        _lowerIntensityToCenter = getIntensityToCenter(imgs[i], lowerSlope, lowerIntercept, lowerXLims)
        lowerIntensityToCenter.append(_lowerIntensityToCenter)

        # find the point along the average intensity curve from getIntensityToCenter
        #  where we are out of the threads and onto the screw body
        # TODO: this will need to change for silhouette images from the jig
        lowerProbableEndOfThreadIndex = None
        lowerIntensityToCenterDiff = np.diff(_lowerIntensityToCenter)
        j = 0
        while (j < len(lowerIntensityToCenterDiff) and lowerProbableEndOfThreadIndex == None):
            if (lowerIntensityToCenterDiff[j] >= 0):
                lowerProbableEndOfThreadIndex = j
            j += 1

        if (lowerProbableEndOfThreadIndex == None):
            raise RuntimeError("Couldn't find inside edge of threads!")

        # we want to use the intensity curve that's 20% of the way between the
        #  initial line and the one identified above as the border between threads
        #  and screw body
        lowerThreadFrequencySampleIndex = int(0.2 * lowerProbableEndOfThreadIndex)

        lowerThreadFrequencySample = getLineOfPixels(imgs[i], lowerSlope, lowerIntercept, -lowerThreadFrequencySampleIndex, lowerXLims)
        lowerThreadFrequencySamples.append(lowerThreadFrequencySample)

        # take the fft of this intensity curve - the strongest response (other
        #  than DC bias) should be the frequency of the threads
        fft = np.abs(np.fft.fft(lowerThreadFrequencySample))
        ffts.append(fft)

        print(str(i+1) + "/" + str(len(imgs)))

    subplotRows = 4
    subplotCols = len(imgs)

    for i in range(len(imgs)):
        plt.subplot(subplotRows, subplotCols, i + 1)
        plt.imshow(imgs[i])
        plt.plot(upperInliers[i][:, 0], upperInliers[i][:, 1], linewidth = 0, marker = '.', color = 'green')
        plt.plot(lowerInliers[i][:, 0], lowerInliers[i][:, 1], linewidth = 0, marker = '.', color = 'red')
        plt.plot(lowerLineY[i], color = 'yellow')
        plt.plot(upperLineY[i], color = 'magenta')
        
        plt.subplot(subplotRows, subplotCols, i + 1 + len(imgs))
        plt.imshow(convs[i] + imgs[i])
        plt.plot(upperInliers[i][:, 0], upperInliers[i][:, 1], linewidth = 0, marker = '.', color = 'green')
        plt.plot(lowerInliers[i][:, 0], lowerInliers[i][:, 1], linewidth = 0, marker = '.', color = 'red')
        plt.plot(lowerLineY[i], color = 'yellow')
        plt.plot(upperLineY[i], color = 'magenta')

        plt.subplot(subplotRows, subplotCols, i + 1 + len(imgs) * 2)
        plt.plot(lowerThreadFrequencySamples[i])

        plt.subplot(subplotRows, subplotCols, i + 1 + len(imgs) * 3)
        freqs = np.fft.fftfreq(len(ffts[i]))
        plt.plot(freqs, ffts[i])

    plt.show()

if __name__ == "__main__":
    main()
