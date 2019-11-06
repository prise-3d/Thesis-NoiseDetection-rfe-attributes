# main imports
import numpy as np
import sys

# image transform imports
from PIL import Image
from skimage import color
from sklearn.decomposition import FastICA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import svd as lin_svd
from scipy.signal import medfilt2d, wiener, cwt
import pywt
import cv2

from ipfml.processing import transform, compression, segmentation
from ipfml import utils

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt


def _get_sobel_filtered_img(lab_img, k_size=3):

    sobelx = cv2.Sobel(lab_img, cv2.CV_64F, 1, 0, ksize=k_size)
    sobely = cv2.Sobel(lab_img, cv2.CV_64F, 0, 1,ksize=k_size)

    sobel_mag = np.array(np.hypot(sobelx, sobely), 'uint8')  # magnitude

    return sobel_mag
    

def _get_26_attributes(lab_img):
    
        img_width, img_height = 200, 200
        arr = np.array(lab_img)

        # compute all filters statistics
        def get_stats(arr, I_filter):

            e1       = np.abs(arr - I_filter)
            L        = np.array(e1)
            mu0      = np.mean(L)
            A        = L - mu0
            H        = A * A
            E        = np.sum(H) / (img_width * img_height)
            P        = np.sqrt(E)

            return mu0, P

        stats = []

        kernel = np.ones((3,3),np.float32)/9
        stats.append(get_stats(arr, cv2.filter2D(arr,-1,kernel)))

        kernel = np.ones((5,5),np.float32)/25
        stats.append(get_stats(arr, cv2.filter2D(arr,-1,kernel)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 0.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 1)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 1.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 0.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 1)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 1.5)))

        stats.append(get_stats(arr, medfilt2d(arr, [3, 3])))

        stats.append(get_stats(arr, medfilt2d(arr, [5, 5])))

        stats.append(get_stats(arr, wiener(arr, [3, 3])))

        stats.append(get_stats(arr, wiener(arr, [5, 5])))

        wave = w2d(arr, 'db1', 2)
        stats.append(get_stats(arr, np.array(wave, 'float64')))

        data = []

        for stat in stats:
            data.append(stat[0])

        for stat in stats:
            data.append(stat[1])

        return data   

def get_image_features(data_type, block):
    """
    Method which returns the data type expected
    """

    data = []

    if 'filters_statistics' in data_type:

        lab_img = transform.get_LAB_L(block)

        data = _get_26_attributes(lab_img)

    if 'filters_statistics_sobel' in data_type:

        lab_img = transform.get_LAB_L(block)

        data = _get_26_attributes(lab_img)

        # add sobel complexity value to array of data (hence we will have 27 attributes)
        sobel_std = np.std(_get_sobel_filtered_img(lab_img))

        data.append(sobel_std)

    if 'svd' in data_type:

        lab_img = transform.get_LAB_L(block)

        data = compression.get_SVD_s(lab_img)

    
    if 'svd_sobel' in data_type:

        lab_img = transform.get_LAB_L(block)
        data = list(compression.get_SVD_s(lab_img))

        sobel_std = np.std(_get_sobel_filtered_img(lab_img))
        data.append(sobel_std)
        
    data = np.array(data) 

    return data


def w2d(arr, mode='haar', level=1):
    #convert to float   
    imArray = arr
    np.divide(imArray, 255)

    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


def _get_mscn_variance(block, sub_block_size=(50, 50)):

    blocks = segmentation.divide_in_blocks(block, sub_block_size)

    data = []

    for block in blocks:
        mscn_coefficients = transform.get_mscn_coefficients(block)
        flat_coeff = mscn_coefficients.flatten()
        data.append(np.var(flat_coeff))

    return np.sort(data)

