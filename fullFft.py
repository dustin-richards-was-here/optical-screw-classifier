"""
An attempt at doing the 2D FFT on the screw images to get the pitch. 

Kind of works a little bit, but mostly not.
"""
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os

# Images to run
img_folder = "img"
files = ["screw1.jpg", "screw2.jpg", "screw3.jpg", "screw4.jpg", "screw5.jpg"]

def fft_image(img):
    # Do the fourier transform
    img_fourier = np.fft.fft2(img)

    # Amplify the magnitude and get the angle
    img_fourier_shifted = np.fft.fftshift(img_fourier)
    img_fourier_mag_disp = 20*np.log10(np.abs(img_fourier_shifted))
    img_fourier_angle_disp = np.angle(img_fourier_shifted)

    return img_fourier_mag_disp, img_fourier_angle_disp
    
def crop_screw(img, pad=0):
    THRESH = 230
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
    return [
        img[head_top-pad:head_bottom+1+pad, screw_start-pad:head_end+1+pad], 
        img[edge_loc_top-pad:edge_loc_bottom+1+pad, head_end-pad:screw_end+1+pad]
        ]

def demo_fft(img_folder, files):
    # Initialize plot
    fig, axs = plt.subplots(len(files), 3)

    # Loop through all images
    for i, file in enumerate(files):
        # Do the FFT
        path = os.path.join(os.getcwd(), img_folder, file)
        img = np.array(ImageOps.grayscale(Image.open(path)))

        screw_head, screw_thread = crop_screw(img)

        mag, angle = fft_image(screw_thread)

        # Add the results to the plot
        axs[i, 0].imshow(img, cmap = 'gray', vmin=0, vmax=255)
        axs[i, 0].set_title("Input Image")
        axs[i, 1].imshow(screw_thread, cmap = 'gray')
        axs[i, 1].set_title("Threads")
        axs[i, 2].imshow(mag, cmap = 'gray')
        axs[i, 2].set_title("FFT Magnitude")

        for ax in axs[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()

demo_fft(img_folder, files)
