
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
sys.path.insert(0, '../..')


def compare_psnr(image1, image2):
    image1 = im.im2uint(image1)
    image2 = im.im2uint(image2)
    psnr_value = psnr(image1, image2)
    return psnr_value


def compare_ssim(image1, image2):
    image1 = im.im2uint(image1)
    image2 = im.im2uint(image2)
    ssim_value = ssim(image1, image2)
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


def plot_images_A2B(A_i, A2B_i, B_i, save_dir, img_path, best_psnr=False, best_ssim=False):
    psnr, ssim = compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
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

    plt.savefig(save_path, bbox_inches='tight')
    plt.tight_layout()
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


def plot_images_A2B_B2A(A_i, A2B_i, B_i, B2A_i, save_dir, epoch):
    psnr_A2B, ssim_A2B = compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
    psnr_B2A, ssim_B2A = compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
    plt.figure(figsize=(10, 5))
    plt.subplot(221)
    plt.imshow(im.dtype.im2uint(A_i.numpy()))
    plt.title('A')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(im.dtype.im2uint(A2B_i.numpy()))
    plt.title(f'A2B\nPSNR: {psnr_A2B:.4f}, SSIM: {ssim_A2B:.4f}')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(im.dtype.im2uint(B_i.numpy()))
    plt.title('B')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(im.dtype.im2uint(B2A_i.numpy()))
    plt.title(f'B2A\nPSNR: {psnr_B2A:.4f}, SSIM: {ssim_B2A:.4f}')
    plt.axis('off')
    plt.tight_layout()
    # save the image with the epoch number only
    plt.savefig(os.path.join(save_dir, f'EPOCH - {epoch}.png'), bbox_inches='tight')
    plt.close()
