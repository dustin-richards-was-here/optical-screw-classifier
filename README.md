The goal of this project is to provide a fast and easy-to-use jig-assisted camera-based rootin-tootin darn-scootin screw classifier.

### threadKernel.py
Uses convolution to locate thread peaks in multiple images.

Usage: `python3 threadKernel.py kernel img1 img2 img3 ...`

### fft.py
Takes a user-provided y value and uses an FFT to identify thread frequency across that slice of the image.