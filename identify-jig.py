from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
import pyransac3d as pyrsc
import math
import os

import pdb

FILE_FROM_CAMERA = "capt0000.jpg"
EMPTY_JIG_IMAGE_FILE = "img/jig-captures/empty.jpg"
# a y value that consistently sits in the middle of the screw. doesn't need to be
#  the exact center.
SCREW_MIDDLE_Y = 1840
MM_PER_PIXEL = 0.01808972504

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
    # for i in range(thresholded.shape[0]):
    #     for j in range(thresholded.shape[1]):
    #         if (thresholded[i,j]):
    #             candidatePixelsList.append([j,i])

    # make arrays out of our pixel lists
    # candidatePixels = np.array(candidatePixelsList)

    candidatePixels = np.flip(np.array(np.where(thresholded)).T, 1)

    # set up the shape to accomodate a z dimension in each entry
    pointsArrayShape = (candidatePixels.shape[0], 3)

    # set the z for all to 0, we're only doing this because the RANSAC library
    #  I picked is designed for 3D
    candidatePoints = np.empty(pointsArrayShape)
    candidatePoints[:, 0:2] = candidatePixels
    candidatePoints[:, 2] = 0

    # run RANSAC on the candidate points
    line = pyrsc.Line()
    slope, intercept, inlierIndices = line.fit(candidatePoints, thresh = 0.01, maxIteration = 500)

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

def getScrewRegions(img, pad=0):
    THRESH = 230 / 255
    img_thresh = img < THRESH
    mid_y = img.shape[0] // 2

    # Find the end of the screw
    # Find the first non-white pixel from right-to-left
    dat_end = []
    for img_row in img_thresh:
        screw_coords = np.argwhere(img_row)
        last_pix = 0
        if len(screw_coords) > 0:
            last_pix = screw_coords[-1,0]
        dat_end.append(last_pix)
    dat_end = np.array(dat_end)

    screw_end = dat_end[mid_y]

    # Find the start of the screw
    screw_start = np.argwhere(img_thresh[mid_y])[0,0]

    # The largest "jump" in end data is (probably) the edge
    diffs = dat_end[1:] - dat_end[:-1]

    # Get the top and bottom of the thread
    edge_loc_top = np.argmax(diffs[:mid_y])
    edge_loc_bottom = mid_y + np.argmin(diffs[mid_y:])+1

    # Go a bit further until past any additional abrupt changes
    orig_edge_loc = edge_loc_top
    while diffs[edge_loc_top+1] > 0.1 * diffs[orig_edge_loc]:
        edge_loc_top -= 1
    orig_edge_loc = edge_loc_bottom
    while diffs[edge_loc_bottom] < 0.1 * diffs[orig_edge_loc-1]:
        edge_loc_bottom += 1

    # Get the start of the thread
    head_end = dat_end[edge_loc_top]

    # Get the top/bottom of the head
    head_top = np.argwhere(dat_end > (screw_start+2))[0,0]
    head_bottom = np.argwhere(dat_end > (screw_start+2))[-1,0]+1

    # Crop and return (head, screw)
    #return [
    #    img[head_top-pad:head_bottom+1+pad, screw_start-pad:head_end+1+pad], 
    #    img[edge_loc_top-pad:edge_loc_bottom+1+pad, head_end-pad:screw_end+1+pad]
    #    ]

    return screw_start, head_end, screw_end

def identify(screwImg: Image, emptyImg: Image, kernelImg: Image, plotParams: dict = None):
    matrix = (0, 0, 0, 0,
              0, 0, 0, 0,
              0, 0, 1, 0)
    
    #img = np.array(ImageOps.grayscale(screwImg)).astype(float) / 255
    img = np.array(screwImg.convert("RGB", matrix).convert("L")).astype(float) / 255
    emptyImg = np.array(emptyImg.convert("RGB", matrix).convert("L")).astype(float) / 255

    img = img - emptyImg

    lowerKernel = np.array(kernelImg).astype(float) / 255
    lowerKernel[int(lowerKernel.shape[0]//2 - 10) : int(lowerKernel.shape[0]//2 - 5), :] = 0
    upperKernel = np.array(kernelImg.rotate(180)).astype(float) / 255

    lowerKernel = (lowerKernel - 0.5) * 2
    upperKernel = (upperKernel - 0.5) * 2

    # convolve the image with the tread kernel and use RANSAC to find a cluster
    #  of points that represents the threadform
    # we do this twice because one of the kernels is upside down in order to
    #  get the threads in the lower half of the image
    lowerConv, lowerInliers, lowerSlope, lowerIntercept = convoleAndRANSAC(img, lowerKernel)
    upperConv, upperInliers, upperSlope, upperIntercept = convoleAndRANSAC(img, upperKernel)

    # combine the two convolution images, really just for visualizing
    conv = lowerConv + upperConv

    # calculate the points of a line for either side of the screw based on the
    #  RANSAC results from above
    x = np.arange(0, img.shape[1])
    lowerLineY = lowerSlope * x + lowerIntercept
    upperLineY = upperSlope * x + upperIntercept

    # find the x coordinates of the furthest left and right inlier point
    upperXLimArgmin = np.argmin(upperInliers[:, 0])
    upperXLimArgmax = np.argmax(upperInliers[:, 0])
    upperXLims = (upperInliers[upperXLimArgmin, 0], upperInliers[upperXLimArgmax, 0])

    # start with a line between these two extreme points. take the average
    #  intensity of the image across this line. step the line's y-intercept
    #  closer to the centerline of the image, take the average intensity.
    #  repeat until at the center.
    upperIntensityToCenter = getIntensityToCenter(img, upperSlope, upperIntercept, upperXLims)

    # find the point along the average intensity curve from getIntensityToCenter
    #  where we are out of the threads and onto the screw body
    # TODO: this will need to change for silhouette images from the jig
    upperProbableEndOfThreadIndex = None
    upperIntensityToCenterDiff = np.diff(upperIntensityToCenter)
    j = 0
    while (j < len(upperIntensityToCenterDiff) and upperProbableEndOfThreadIndex == None):
        if (upperIntensityToCenterDiff[j] >= 0):
            upperProbableEndOfThreadIndex = j
        j += 1

    if (upperProbableEndOfThreadIndex == None):
        raise RuntimeError("Couldn't find inside edge of threads!")

    # we want to use the intensity curve that's 20% of the way between the
    #  initial line and the one identified above as the border between threads
    #  and screw body
    upperThreadFrequencySampleIndex = int(0.2 * upperProbableEndOfThreadIndex)

    #upperThreadFrequencySample = getLineOfPixels(img, upperSlope, upperIntercept, -upperThreadFrequencySampleIndex, upperXLims)
    upperThreadFrequencySample = getLineOfPixels(img, upperSlope, upperIntercept, 0, upperXLims)

    # take the fft of this intensity curve - the strongest response (other
    #  than DC bias) should be the frequency of the threads
    fft = np.abs(np.fft.fft(upperThreadFrequencySample))

    # use the upper and upper y-intercepts from RANSAC to determine the
    #  screw diameter
    diameterPixels = abs(lowerIntercept - upperIntercept)

    # find the largest peak in the FFT (other than the DC bias in the center)
    #  and find the period in pixels that corresponds to it, that's our
    #  thread pitch
    freqs = np.fft.fftfreq(len(fft))
    maxMagnitudeIndex = np.argmax(fft[int(len(fft) / 2 + 1) : len(fft)]) + int(len(fft) / 2 + 1)
    threadPitchFrequency = abs(freqs[maxMagnitudeIndex])
    threadPitchPixels = 1 / threadPitchFrequency

    headStartX, headEndX, threadsEndX = getScrewRegions(img)
    headLengthPixels = abs(headStartX - headEndX)
    threadLengthPixels = abs(headEndX - threadsEndX)

    if (plotParams):
        subplotRows = 4

        subplotCols = None
        if (plotParams['stream'] == True):
            subplotCols = 1
        else:
            subplotCols = plotParams['numImages']

        imageIndex = plotParams['imageIndex']

        plt.subplot(subplotRows, subplotCols, imageIndex + 1)
        plt.imshow(img + conv)
        plt.plot(upperInliers[:, 0], upperInliers[:, 1], linewidth = 0, marker = '.', color = 'green')
        plt.plot(lowerInliers[:, 0], lowerInliers[:, 1], linewidth = 0, marker = '.', color = 'red')
        
        plt.subplot(subplotRows, subplotCols, imageIndex + 1 + subplotCols)
        plt.imshow(img)
        plt.plot(lowerLineY, color = 'yellow')
        plt.plot(upperLineY, color = 'magenta')
        plt.plot((headStartX, headStartX), (0, img.shape[0]), color = 'red')
        plt.plot((headEndX, headEndX), (0, img.shape[0]), color = 'red')
        plt.plot((threadsEndX, threadsEndX), (0, img.shape[0]), color = 'red')

        plt.subplot(subplotRows, subplotCols, imageIndex + 1 + subplotCols * 2)
        plt.plot(upperThreadFrequencySample)

        plt.subplot(subplotRows, subplotCols, imageIndex + 1 + subplotCols * 3)
        plt.plot(freqs, fft)

        if (plotParams['stream'] == True):
            plt.show()

    return diameterPixels, threadLengthPixels, threadPitchPixels

def printUsage():
    print("Usage (images): " + sys.argv[0] + " kernel img1 img2 img3 ...")
    print("Usage (camera): " + sys.argv[0] + " kernel stream")

def main():
    if (len(sys.argv) < 3):
        printUsage()
        return

    kernelImgFile = sys.argv[1]
    kernelImg = ImageOps.grayscale(Image.open(kernelImgFile))

    emptyImg = Image.open(EMPTY_JIG_IMAGE_FILE)
    emptyImg = emptyImg.crop((0, SCREW_MIDDLE_Y - emptyImg.size[1]/6, emptyImg.size[0]/3*2, SCREW_MIDDLE_Y + emptyImg.size[1]/6))

    numImages = 0
    stream = False
    if (sys.argv[2] == "stream"):
        numImages = 999999
        stream = True
    else:
        numImages = len(sys.argv) - 2

    imgs = []
    plotParams = {'numImages': numImages}

    if (stream):
        plotParams['stream'] = True
    else:
        plotParams['stream'] = False

    for i in range(numImages):
        file = None
        if (stream):
            file = FILE_FROM_CAMERA
            try:
                os.remove(file)
            except:
                pass
            os.system("gphoto2 --capture-image-and-download 1> /dev/null")
            plotParams['imageIndex'] = 0
        else:
            file = sys.argv[i+2]
            plotParams['imageIndex'] = i

        img = Image.open(file)

        #if (stream):
        #    # img = ImageOps.scale(img, 0.25)
        #    img = img.crop((0, img.size[1]/4, img.size[0]/3*2, img.size[1]/4*3))
        #    plt.imshow(img)
        #    plt.show()

        #img = img.crop((0, img.size[1]/4, img.size[0]/3*2, img.size[1]/4*3))
        img = img.crop((0, SCREW_MIDDLE_Y - img.size[1]/6, img.size[0]/3*2, SCREW_MIDDLE_Y + img.size[1]/6))

        diameterPixels, threadLengthPixels, threadPitchPixels = identify(img, emptyImg, kernelImg, plotParams)
        print("===================")
        print("Image " + str(i + 1))
        print("Length (mm): " + str(threadLengthPixels * MM_PER_PIXEL))
        print("Diameter (mm): " + str(diameterPixels * MM_PER_PIXEL))
        print("Pitch (mm): " + str(threadPitchPixels * MM_PER_PIXEL))
        print("===================")

    plt.show()

if __name__ == "__main__":
    main()
