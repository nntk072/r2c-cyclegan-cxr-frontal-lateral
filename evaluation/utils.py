
# Return PSNR and SSIM values of two images
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from math import log10, sqrt
import imlib as im
import os
import sys
import scipy.io
import scipy
import torch
import samscore
sys.path.insert(0, '../..')

SAMScore_Evaluation = samscore.SAMScore(model_type = "vit_l" )
def compare_psnr(image1, image2):
    image1 = im.im2uint(image1)
    image2 = im.im2uint(image2)
    # psnr_value = psnr(image1, image2)
    # Setting up the data range from 0 to 255
    psnr_value = psnr(image1, image2, data_range=255)
    return psnr_value

def compare_psnr_mathematical(image1, image2):
    image1 = im.im2uint(image1)
    image2 = im.im2uint(image2)
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr_value = 20 * log10(max_pixel / sqrt(mse))
    # Converting to numpy.float64
    psnr_value = np.float64(psnr_value)
    return psnr_value

def compare_ssim(image1, image2):
    image1 = im.im2uint(image1)
    image2 = im.im2uint(image2)

    # ssim_value = ssim(image1, image2)
    # Setting up the data range from 0 to 255
    ssim_value = ssim(image1, image2, data_range=255)
    
    # # Do not use the library for calculating
    # # The value of C1 and C2 are from the paper
    # C1 = (0.01 * 255) ** 2
    # C2 = (0.03 * 255) ** 2
    # # Mean of the images
    # mu1 = np.mean(image1)
    # mu2 = np.mean(image2)
    # # Variance of the images
    # sigma1 = np.var(image1)
    # sigma2 = np.var(image2)
    # # Covariance of the images
    # sigma12 = np.mean((image1 - mu1) * (image2 - mu2))
    # # Calculate the SSIM value
    # ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    
    # Save into mat file for 2 images for checking
    # scipy.io.savemat('image1_ssim.mat', {'image1': image1})
    # scipy.io.savemat('image2_ssim.mat', {'image2': image2})
    
    # img1 = image1.astype(np.float64)
    # img2 = image2.astype(np.float64)
    # size = 11
    # sigma = 1.5
    # x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    # g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    # window = g/g.sum()
    # K1 = 0.01
    # K2 = 0.03
    # L = 255 #bitdepth of image
    # C1 = (K1*L)**2
    # C2 = (K2*L)**2
    # mu1 = signal.fftconvolve(window, img1, mode='valid')
    # mu2 = signal.fftconvolve(window, img2, mode='valid')
    # mu1_sq = mu1*mu1
    # mu2_sq = mu2*mu2
    # mu1_mu2 = mu1*mu2
    # sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    # sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    # sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    # cs_map = False
    # if cs_map:
    #     return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
    #                 (sigma1_sq + sigma2_sq + C2)), 
    #             (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    # else:
    #     ssim_ndarray = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
    #                 (sigma1_sq + sigma2_sq + C2))
    #     return ssim_ndarray.mean()
    return ssim_value

def compare_ssim_mathematical(image1, image2):
    image1 = im.im2uint(image1)
    image2 = im.im2uint(image2)
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = np.mean(image1)
    mu2 = np.mean(image2)
    sigma1 = np.var(image1)
    sigma2 = np.var(image2)
    sigma12 = np.mean((image1 - mu1) * (image2 - mu2))
    ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    )
    # Converting to numpy.float64
    ssim_value = np.float64(ssim_value)
    return ssim_value

def compute_psnr_ssim(img1, img2):
    # Check if the images are in the correct format
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    psnr = compare_psnr(img1, img2)
    ssim = compare_ssim(img1, img2)
    return psnr, ssim

def compute_psnr_ssim_mathematical(img1, img2):
    # Check if the images are in the correct format
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    psnr = compare_psnr_mathematical(img1, img2)
    ssim = compare_ssim_mathematical(img1, img2)
    return psnr, ssim

def sams_score(img1, img2, SAMScore_Evaluation):
    # source = torch.from_numpy(img1.transpose(2,0,1)).unsqueeze(0).float()
    # source = torch.cat((source, source, source), dim=0)
    
    # target = torch.from_numpy(img2.transpose(2,0,1)).unsqueeze(0).float()
    # target = torch.cat((target, target, target), dim=0)
    
    # sams = SAMScore_Evaluation.evaluation_from_torch(source, target)
    # # Converting from Tensor.__format__ to float
    # sams = sams.item()
    # print(sams.type())
    
    # Using CUDA
    source = torch.from_numpy(img2.transpose(2,0,1)).unsqueeze(0).float()
    source = torch.cat((source, source, source), dim=0)
    

    target = torch.from_numpy(img1.transpose(2,0,1)).unsqueeze(0).float()
    target = torch.cat((target, target, target), dim=0)
    
    sams = SAMScore_Evaluation.evaluation_from_torch(source, target)
    # sams now is tensor([0.9875, 0.9875, 0.9875]), take 1 value only
    sams = sams[0].item()
    return sams



def plot_images_A2B(A_i, A2B_i, B_i, save_dir, img_path, best_psnr=False, best_ssim=False):
    psnr, ssim = compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
    # sams_score = sams_score(A2B_i.numpy(), B_i.numpy())
    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.imshow(im.dtype.im2uint(A_i.numpy()))
    plt.title('A')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(im.dtype.im2uint(A2B_i.numpy()))
    plt.title(f'A2B\nPSNR: {psnr:.4f}, SSIM: {ssim:.4f}')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(im.dtype.im2uint(B_i.numpy()))
    plt.title('B')
    plt.axis('off')
    # Get the image name and extension
    img_name = os.path.basename(img_path)

    # Join the save directory and the image name
    if best_psnr and best_ssim:
        save_path = os.path.join(save_dir, f'best_psnr_ssim')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return psnr, ssim

    if best_psnr:
        save_path = os.path.join(save_dir, f'best_psnr')
    elif best_ssim:
        save_path = os.path.join(save_dir, f'best_ssim')
    else:
        save_path = os.path.join(save_dir, img_name)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return psnr, ssim


def plot_images_B2A(B_i, B2A_i, A_i, save_dir, img_path, best_psnr=False, best_ssim=False):
    psnr, ssim = compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
    # sams_score = sams_score(B2A_i.numpy(), A_i.numpy())
    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.imshow(im.dtype.im2uint(B_i.numpy()))
    plt.title('B')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(im.dtype.im2uint(B2A_i.numpy()))
    # Change this into B2A, psnr, ssim
    # plt.title('B2A')
    plt.title(f'B2A\nPSNR: {psnr:.4f}, SSIM: {ssim:.4f}')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(im.dtype.im2uint(A_i.numpy()))
    plt.title('A')
    plt.axis('off')
    # Get the image name and extension
    img_name = os.path.basename(img_path)
    # Join the save directory and the image name
    # save_path = os.path.join(save_dir, img_name)
    if best_psnr and best_ssim:
        save_path = os.path.join(save_dir, f'best_psnr_ssim')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return psnr, ssim

    if best_psnr:
        save_path = os.path.join(save_dir, f'best_psnr')
    elif best_ssim:
        save_path = os.path.join(save_dir, f'best_ssim')
    else:
        save_path = os.path.join(save_dir, img_name)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return psnr, ssim


def plot_all_images_A2B(A_i, A2B_list, B_i, psnr_list, ssim_list, save_dir, img_path, method_list):
    plt.figure(figsize=(15, 4))
    A2B_list_length = len(A2B_list)
    total_plot = 1 + A2B_list_length + 1

    plt.subplot(1, total_plot, 1)
    plt.imshow(im.dtype.im2uint(A_i.numpy()))
    plt.title('A')
    plt.axis('off')

    for i in range(A2B_list_length):
        plt.subplot(1, total_plot, i+2)
        plt.imshow(im.dtype.im2uint(A2B_list[i].numpy()))
        # title_text = f'PSNR: {psnr_list[i]:.4f}\nSSIM: {ssim_list[i]:.4f}\nMethod: {method_list[i]}'
        title_text = f'PSNR: {psnr_list[i]:.4f}\nSSIM: {ssim_list[i]:.4f}'
        plt.title(title_text)  # Adjusted title text
        plt.axis('off')

    plt.subplot(1, total_plot, total_plot)
    plt.imshow(im.dtype.im2uint(B_i.numpy()))
    plt.title('B')
    plt.axis('off')

    img_name = os.path.basename(img_path)
    save_path = os.path.join(save_dir, img_name)

    # Use tight_layout before saving figure to avoid overlap
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_all_images_B2A(B_i, B2A_list, A_i, psnr_list, ssim_list, save_dir, img_path, method_list):
    plt.figure(figsize=(15, 4))
    B2A_list_length = len(B2A_list)
    total_plot = 1 + B2A_list_length + 1
    plt.subplot(1, total_plot, 1)
    plt.imshow(im.dtype.im2uint(B_i.numpy()))
    plt.title('B')
    plt.axis('off')
    for i in range(B2A_list_length):
        plt.subplot(1, total_plot, i+2)
        plt.imshow(im.dtype.im2uint(B2A_list[i].numpy()))
        # title_text = f'PSNR: {psnr_list[i]:.4f}\nSSIM: {ssim_list[i]:.4f}\nMethod: {method_list[i]}'
        # Do not need method
        title_text = f'PSNR: {psnr_list[i]:.4f}\nSSIM: {ssim_list[i]:.4f}'
        plt.title(title_text)  # Adjusted title text
        plt.axis('off')
    plt.subplot(1, total_plot, total_plot)
    plt.imshow(im.dtype.im2uint(A_i.numpy()))
    plt.title('A')
    plt.axis('off')
    img_name = os.path.basename(img_path)
    save_path = os.path.join(save_dir, img_name)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Same idea of plot_images_A2B and B2A, but combine them in one function


def plot_images_A2B_B2A(A_i, A2B_i, B_i, B2A_i, save_dir, epoch, SAMScore_Evaluation = None):
    psnr_A2B, ssim_A2B = compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
    psnr_B2A, ssim_B2A = compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
    # sams_score_A2B = sams_score(A2B_i.numpy(), B_i.numpy(), SAMScore_Evaluation)
    # sams_score_B2A = sams_score(B2A_i.numpy(), A_i.numpy(), SAMScore_Evaluation)
    plt.figure(figsize=(10, 5))
    plt.subplot(221)
    plt.imshow(im.dtype.im2uint(A_i.numpy()))
    plt.title('A')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(im.dtype.im2uint(A2B_i.numpy()))
    # plt.title(f'A2B\nPSNR: {psnr_A2B:.4f}, SSIM: {ssim_A2B:.4f}, SAMScore: {sams_score_A2B:.4f}')
    plt.title(f'A2B\nPSNR: {psnr_A2B:.4f}, SSIM: {ssim_A2B:.4f}')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(im.dtype.im2uint(B_i.numpy()))
    plt.title('B')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(im.dtype.im2uint(B2A_i.numpy()))
    # plt.title(f'B2A\nPSNR: {psnr_B2A:.4f}, SSIM: {ssim_B2A:.4f}, SAMScore: {sams_score_B2A:.4f}')
    plt.title(f'B2A\nPSNR: {psnr_B2A:.4f}, SSIM: {ssim_B2A:.4f}')
    plt.axis('off')
    plt.tight_layout()
    # save the image with the epoch number only
    plt.savefig(os.path.join(save_dir, f'EPOCH - {epoch}.png'), bbox_inches='tight')
    plt.close()
