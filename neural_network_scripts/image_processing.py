"""
image-processing.py
~~~~~~~~~~

Sourced from: https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
"""


#### Libraries
# Standard library
import math

# Third-party libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def process_image(file_path):

    # read the image
    gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # rescale it to 28px by 28px and invert it (black background)
    gray = cv2.resize(255-gray, (28, 28))
    
    # better black and white version
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    ### Size normal images to fit in 20px by 20px box
    # remove every row and column at the sides of the image which are completely black
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape
    
    # Resize outer box to fit into a 20px by 20px box
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))
        
    # Pad missing black rows and columns to add 0s to the sides until image is 
    # 28px by 28px
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    
    # Shift inner 20px by 20px box to be centered using center of mass
    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted
    

    # # save the processed images
    # cv2.imwrite("processed_handwritten_2.png", gray)

    """
    we need to store the flatten image and generate
    the correct_vals array
    correct_val for the first digit (9) would be
    [0,0,0,0,0,0,0,0,0,1]
    """
    # images = [flatten]
    # correct_val = np.zeros((10))
    # correct_val[2] = 1
    
    return gray


def print_image(image, ax=None):
    
    if image.shape == (784, 1):
        image = image.reshape(28, 28)
    
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    