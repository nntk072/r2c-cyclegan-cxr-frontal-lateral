
# Return PSNR and SSIM values of two images
from math import log10, sqrt
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_psnr(image1, image2):
    # convert images to grayscale
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # calculate mse
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    # calculate psnr
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def compare_ssim(image1, image2):
    # convert images to grayscale
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # calculate ssim
    ssim_value = ssim(image1, image2, full=true)
    return ssim_value

def compute_psnr_ssim(img1, img2):
    psnr = compare_psnr(img1, img2)
    ssim = compare_ssim(img1, img2, multichannel=True)
    return psnr, ssim
