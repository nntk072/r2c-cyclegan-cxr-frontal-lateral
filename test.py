import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl
import matplotlib.pyplot as plt

import sys
import data
import module
import os
import evaluation as ev
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir', default='output/')
py.arg('--batch_size', type=int, default=32)
py.arg('--method', help='convolutional, operational, unet, anotherunet, operational_unet',
       default='convolutional')
py.arg('--loss_method', help='none, adversarial, total, cycle, generator, discriminator, identity, all',
       default='adversarial')
py.arg('--evaluation_method', help='none, psnr, ssim, both', default='none')
test_args = py.args()
args = py.args_from_yaml(
    py.join(test_args.experiment_dir, test_args.method, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = py.glob(
    py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
B_img_paths_test = py.glob(
    py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')
A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
B_dataset_test = data.make_dataset(B_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
method = args.method
# model
if args.method == 'convolutional':
    G_A2B = module.ResnetGenerator(
        input_shape=(args.crop_size, args.crop_size, 3))
    G_B2A = module.ResnetGenerator(
        input_shape=(args.crop_size, args.crop_size, 3))
elif args.method == 'operational':
    G_A2B = module.OpGenerator(input_shape=(
        args.crop_size, args.crop_size, 3), q=args.q)
    G_B2A = module.OpGenerator(input_shape=(
        args.crop_size, args.crop_size, 3), q=args.q)
elif args.method == 'unet':
    G_A2B = module.UNetGenerator(
        input_shape=(args.crop_size, args.crop_size, 3))
    G_B2A = module.UNetGenerator(
        input_shape=(args.crop_size, args.crop_size, 3))
elif args.method == 'anotherunet':
    G_A2B = module.AnotherUnetGenerator(
        input_shape=(args.crop_size, args.crop_size, 3))
    G_B2A = module.AnotherUnetGenerator(
        input_shape=(args.crop_size, args.crop_size, 3))
elif args.method == 'operational_unet':
    G_A2B = module.OpUnetGenerator(
        input_shape=(args.crop_size, args.crop_size, 3), q=args.q)
    G_B2A = module.OpUnetGenerator(
        input_shape=(args.crop_size, args.crop_size, 3), q=args.q)
else:
    raise NotImplementedError
A2B_g_loss_list = np.array([])
B2A_g_loss_list = np.array([])
A2B2A_cycle_loss_list = np.array([])
B2A2B_cycle_loss_list = np.array([])
A2A_id_loss_list = np.array([])
B2B_id_loss_list = np.array([])
A_d_loss_list = np.array([])
B_d_loss_list = np.array([])
ssim_A2B_list = np.array([])
psnr_A2B_list = np.array([])
ssim_B2A_list = np.array([])
psnr_B2A_list = np.array([])
ep_list = np.array([])
for ep in range(0, 1000):
    A2B_g_loss = np.load(
        f'output/{method}/plot_data/training/loss_A2B_g_loss_{ep}.npy')
    B2A_g_loss = np.load(
        f'output/{method}/plot_data/training/loss_B2A_g_loss_{ep}.npy')
    A2B2A_cycle_loss = np.load(
        f'output/{method}/plot_data/training/loss_A2B2A_cycle_loss_{ep}.npy')
    B2A2B_cycle_loss = np.load(
        f'output/{method}/plot_data/training/loss_B2A2B_cycle_loss_{ep}.npy')
    A2A_id_loss = np.load(
        f'output/{method}/plot_data/training/loss_A2A_id_loss_{ep}.npy')
    B2B_id_loss = np.load(
        f'output/{method}/plot_data/training/loss_B2B_id_loss_{ep}.npy')
    A_d_loss = np.load(
        f'output/{method}/plot_data/training/loss_A_d_loss_{ep}.npy')
    B_d_loss = np.load(
        f'output/{method}/plot_data/training/loss_B_d_loss_{ep}.npy')
    ssim_A2B = np.load(
        f'output/{method}/plot_data/training/ssim_A2B_value_list_{ep}.npy')
    psnr_A2B = np.load(
        f'output/{method}/plot_data/training/psnr_A2B_value_list_{ep}.npy')
    ssim_B2A = np.load(
        f'output/{method}/plot_data/training/ssim_B2A_value_list_{ep}.npy')
    psnr_B2A = np.load(
        f'output/{method}/plot_data/training/psnr_B2A_value_list_{ep}.npy')
    iterations = np.load(
        f'output/{method}/plot_data/training/iterations_{ep}.npy')

    # Calculate the mean of the loss data for each iteration and save into the list
    A2B_g_loss_list = np.append(A2B_g_loss_list, np.mean(A2B_g_loss))
    B2A_g_loss_list = np.append(B2A_g_loss_list, np.mean(B2A_g_loss))
    A2B2A_cycle_loss_list = np.append(
        A2B2A_cycle_loss_list, np.mean(A2B2A_cycle_loss))
    B2A2B_cycle_loss_list = np.append(
        B2A2B_cycle_loss_list, np.mean(B2A2B_cycle_loss))
    A2A_id_loss_list = np.append(A2A_id_loss_list, np.mean(A2A_id_loss))
    B2B_id_loss_list = np.append(B2B_id_loss_list, np.mean(B2B_id_loss))
    A_d_loss_list = np.append(A_d_loss_list, np.mean(A_d_loss))
    B_d_loss_list = np.append(B_d_loss_list, np.mean(B_d_loss))
    ssim_A2B_list = np.append(ssim_A2B_list, np.mean(ssim_A2B))
    psnr_A2B_list = np.append(psnr_A2B_list, np.mean(psnr_A2B))
    ssim_B2A_list = np.append(ssim_B2A_list, np.mean(ssim_B2A))
    psnr_B2A_list = np.append(psnr_B2A_list, np.mean(psnr_B2A))
    ep_list = np.append(ep_list, ep)


# Do the same with the valid
A2B_g_loss_list_valid = np.array([])
B2A_g_loss_list_valid = np.array([])
A2B2A_cycle_loss_list_valid = np.array([])
B2A2B_cycle_loss_list_valid = np.array([])
A2A_id_loss_list_valid = np.array([])
B2B_id_loss_list_valid = np.array([])
A_d_loss_list_valid = np.array([])
B_d_loss_list_valid = np.array([])
ssim_A2B_list_valid = np.array([])
psnr_A2B_list_valid = np.array([])
ssim_B2A_list_valid = np.array([])
psnr_B2A_list_valid = np.array([])

ep_list_valid = np.array([])
for ep in range(0, 1000):  # the name of the folder is validation
    A2B_g_loss = np.load(
        f'output/{method}/plot_data/validation/loss_A2B_g_loss_{ep}.npy')
    B2A_g_loss = np.load(
        f'output/{method}/plot_data/validation/loss_B2A_g_loss_{ep}.npy')
    A2B2A_cycle_loss = np.load(
        f'output/{method}/plot_data/validation/loss_A2B2A_cycle_loss_{ep}.npy')
    B2A2B_cycle_loss = np.load(
        f'output/{method}/plot_data/validation/loss_B2A2B_cycle_loss_{ep}.npy')
    A2A_id_loss = np.load(
        f'output/{method}/plot_data/validation/loss_A2A_id_loss_{ep}.npy')
    B2B_id_loss = np.load(
        f'output/{method}/plot_data/validation/loss_B2B_id_loss_{ep}.npy')
    A_d_loss = np.load(
        f'output/{method}/plot_data/validation/loss_A_d_loss_{ep}.npy')
    B_d_loss = np.load(
        f'output/{method}/plot_data/validation/loss_B_d_loss_{ep}.npy')
    ssim_A2B = np.load(
        f'output/{method}/plot_data/validation/ssim_A2B_value_list_{ep}.npy')
    psnr_A2B = np.load(
        f'output/{method}/plot_data/validation/psnr_A2B_value_list_{ep}.npy')
    ssim_B2A = np.load(
        f'output/{method}/plot_data/validation/ssim_B2A_value_list_{ep}.npy')
    psnr_B2A = np.load(
        f'output/{method}/plot_data/validation/psnr_B2A_value_list_{ep}.npy')
    iterations = np.load(
        f'output/{method}/plot_data/validation/iterations_{ep}.npy')

    # Calculate the mean of the loss data for each iteration and save into the list
    A2B_g_loss_list_valid = np.append(
        A2B_g_loss_list_valid, np.mean(A2B_g_loss))
    B2A_g_loss_list_valid = np.append(
        B2A_g_loss_list_valid, np.mean(B2A_g_loss))
    A2B2A_cycle_loss_list_valid = np.append(
        A2B2A_cycle_loss_list_valid, np.mean(A2B2A_cycle_loss))
    B2A2B_cycle_loss_list_valid = np.append(
        B2A2B_cycle_loss_list_valid, np.mean(B2A2B_cycle_loss))
    A2A_id_loss_list_valid = np.append(
        A2A_id_loss_list_valid, np.mean(A2A_id_loss))
    B2B_id_loss_list_valid = np.append(
        B2B_id_loss_list_valid, np.mean(B2B_id_loss))
    A_d_loss_list_valid = np.append(A_d_loss_list_valid, np.mean(A_d_loss))
    B_d_loss_list_valid = np.append(B_d_loss_list_valid, np.mean(B_d_loss))
    ssim_A2B_list_valid = np.append(ssim_A2B_list_valid, np.mean(ssim_A2B))
    psnr_A2B_list_valid = np.append(psnr_A2B_list_valid, np.mean(psnr_A2B))
    ssim_B2A_list_valid = np.append(ssim_B2A_list_valid, np.mean(ssim_B2A))
    psnr_B2A_list_valid = np.append(psnr_B2A_list_valid, np.mean(psnr_B2A))
    ep_list_valid = np.append(ep_list_valid, ep)

# Calculate the adversarial loss
A2B_adversarial_loss = np.add(A2B_g_loss_list, A_d_loss_list)
B2A_adversarial_loss = np.add(B2A_g_loss_list, B_d_loss_list)
A2B_adversarial_loss_valid = np.add(A2B_g_loss_list_valid, A_d_loss_list_valid)
B2A_adversarial_loss_valid = np.add(B2A_g_loss_list_valid, B_d_loss_list_valid)

# Calculate the total loss, remember tot ake the cycle_loss_weight from args
A2B_total_loss = np.add(A2B_adversarial_loss,
                        np.multiply(A2B2A_cycle_loss_list, args.cycle_loss_weight))
B2A_total_loss = np.add(B2A_adversarial_loss,
                        np.multiply(B2A2B_cycle_loss_list, args.cycle_loss_weight))
A2B_total_loss_valid = np.add(A2B_adversarial_loss_valid,
                              np.multiply(A2B2A_cycle_loss_list_valid, args.cycle_loss_weight))
B2A_total_loss_valid = np.add(B2A_adversarial_loss_valid,
                              np.multiply(B2A2B_cycle_loss_list_valid, args.cycle_loss_weight))

# Plot the Adversarial and Total loss in one plot with the same format as
plt.figure(figsize=(20, 12))  # Increase figure size
plt.plot(ep_list, A2B_adversarial_loss,
         label='A2B_adversarial_loss', linewidth=1)
plt.plot(ep_list, B2A_adversarial_loss,
         label='B2A_adversarial_loss', linewidth=1)
plt.plot(ep_list_valid, A2B_adversarial_loss_valid,
         label='A2B_adversarial_loss_valid', linewidth=1)
plt.plot(ep_list_valid, B2A_adversarial_loss_valid,
         label='B2A_adversarial_loss_valid', linewidth=1)
plt.legend(fontsize='large')
plt.title('Adversarial Losses', fontsize='x-large')
plt.xlabel('Epochs', fontsize='large')
plt.ylabel('Loss', fontsize='large')
plt.grid(which='major', color='black', linewidth=0.5)
plt.xlim(ep_list[0], ep_list[-1])
plt.xticks(np.arange(ep_list[0], ep_list[-1], 50))
if max(max(A2B_adversarial_loss), max(B2A_adversarial_loss), max(A2B_adversarial_loss_valid), max(B2A_adversarial_loss_valid)) < 0.25:
    plt.ylim(0, 0.25)
    plt.yticks(np.arange(0, 0.25, 0.025))
elif max(max(A2B_adversarial_loss), max(B2A_adversarial_loss), max(A2B_adversarial_loss_valid), max(B2A_adversarial_loss_valid)) < 0.5:
    plt.ylim(0, 0.5)
    plt.yticks(np.arange(0, 0.5, 0.05))
elif max(max(A2B_adversarial_loss), max(B2A_adversarial_loss), max(A2B_adversarial_loss_valid), max(B2A_adversarial_loss_valid)) < 0.75:
    plt.ylim(0, 0.75)
    plt.yticks(np.arange(0, 0.75, 0.05))
elif max(max(A2B_adversarial_loss), max(B2A_adversarial_loss), max(A2B_adversarial_loss_valid), max(B2A_adversarial_loss_valid)) < 1:
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, 0.05))
elif max(max(A2B_adversarial_loss), max(B2A_adversarial_loss), max(A2B_adversarial_loss_valid), max(B2A_adversarial_loss_valid)) > 4:
    plt.ylim(0, 4)
    plt.yticks(np.arange(0, 4, 0.2))
else:
    plt.ylim(0, max(max(A2B_adversarial_loss), max(B2A_adversarial_loss),
                    max(A2B_adversarial_loss_valid), max(B2A_adversarial_loss_valid)))
    plt.yticks(np.arange(0, max(max(A2B_adversarial_loss), max(B2A_adversarial_loss), max(
        A2B_adversarial_loss_valid), max(B2A_adversarial_loss_valid)), 0.1))
# Save as high-res image
plt.savefig(f'output/{method}/adversarial_losses.png')
plt.close()

plt.figure(figsize=(20, 12))  # Increase figure size
plt.plot(ep_list, A2B_total_loss, label='A2B_total_loss', linewidth=1)
plt.plot(ep_list, B2A_total_loss, label='B2A_total_loss', linewidth=1)
plt.plot(ep_list_valid, A2B_total_loss_valid,
         label='A2B_total_loss_valid', linewidth=1)
plt.plot(ep_list_valid, B2A_total_loss_valid,
         label='B2A_total_loss_valid', linewidth=1)
plt.legend(fontsize='large')
plt.title('Total Losses', fontsize='x-large')
plt.xlabel('Epochs', fontsize='large')
plt.ylabel('Loss', fontsize='large')
plt.grid(which='major', color='black', linewidth=0.5)
plt.xlim(ep_list[0], ep_list[-1])
plt.xticks(np.arange(ep_list[0], ep_list[-1], 50))
if max(max(A2B_total_loss), max(B2A_total_loss), max(A2B_total_loss_valid), max(B2A_total_loss_valid)) < 0.25:
    plt.ylim(0, 0.25)
    plt.yticks(np.arange(0, 0.25, 0.025))
elif max(max(A2B_total_loss), max(B2A_total_loss), max(A2B_total_loss_valid), max(B2A_total_loss_valid)) < 0.5:
    plt.ylim(0, 0.5)
    plt.yticks(np.arange(0, 0.5, 0.05))
elif max(max(A2B_total_loss), max(B2A_total_loss), max(A2B_total_loss_valid), max(B2A_total_loss_valid)) < 0.75:
    plt.ylim(0, 0.75)
    plt.yticks(np.arange(0, 0.75, 0.05))
elif max(max(A2B_total_loss), max(B2A_total_loss), max(A2B_total_loss_valid), max(B2A_total_loss_valid)) < 1:
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, 0.05))
# Limit into 4
elif max(max(A2B_total_loss), max(B2A_total_loss), max(A2B_total_loss_valid), max(B2A_total_loss_valid)) > 4:
    plt.ylim(0, 4)
    plt.yticks(np.arange(0, 4, 0.2))
else:
    plt.ylim(0, max(max(A2B_total_loss), max(B2A_total_loss),
                    max(A2B_total_loss_valid), max(B2A_total_loss_valid)))
    plt.yticks(np.arange(0, max(max(A2B_total_loss), max(B2A_total_loss), max(
        A2B_total_loss_valid), max(B2A_total_loss_valid)), 0.2))
plt.savefig(f'output/{method}/total_losses.png')  # Save as high-res image
plt.close()

# Choosing the epoch with the lowest adversarial loss valid to plot the result
lowest_A2B_adversarial_loss_valid = np.min(A2B_adversarial_loss_valid)
lowest_B2A_adversarial_loss_valid = np.min(B2A_adversarial_loss_valid)
lowest_A2B_adversarial_loss_valid_index = np.where(
    A2B_adversarial_loss_valid == lowest_A2B_adversarial_loss_valid)
lowest_B2A_adversarial_loss_valid_index = np.where(
    B2A_adversarial_loss_valid == lowest_B2A_adversarial_loss_valid)

# Choosing the epoch with the lowest total loss valid to plot the result
lowest_A2B_total_loss_valid = np.min(A2B_total_loss_valid)
lowest_B2A_total_loss_valid = np.min(B2A_total_loss_valid)
lowest_A2B_total_loss_valid_index = np.where(
    A2B_total_loss_valid == lowest_A2B_total_loss_valid)
lowest_B2A_total_loss_valid_index = np.where(
    B2A_total_loss_valid == lowest_B2A_total_loss_valid)

# Choosing the epoch with the lowest cycle loss valid to plot the result
lowest_A2B_cycle_loss_valid = np.min(A2B2A_cycle_loss_list_valid)
lowest_B2A_cycle_loss_valid = np.min(B2A2B_cycle_loss_list_valid)
lowest_A2B_cycle_loss_valid_index = np.where(
    A2B2A_cycle_loss_list_valid == lowest_A2B_cycle_loss_valid)
lowest_B2A_cycle_loss_valid_index = np.where(
    B2A2B_cycle_loss_list_valid == lowest_B2A_cycle_loss_valid)

# Choosing the epoch with the lowest generator loss valid to plot the result
lowest_A2B_g_loss_valid = np.min(A2B_g_loss_list_valid)
lowest_B2A_g_loss_valid = np.min(B2A_g_loss_list_valid)
lowest_A2B_g_loss_list_valid_index = np.where(
    A2B_g_loss_list_valid == lowest_A2B_g_loss_valid)
lowest_B2A_g_loss_list_valid_index = np.where(
    B2A_g_loss_list_valid == lowest_B2A_g_loss_valid)

# Choosing the epoch with the lowest discriminator loss valid to plot the result
lowest_A2B_d_loss_valid = np.min(A_d_loss_list_valid)
lowest_B2A_d_loss_valid = np.min(B_d_loss_list_valid)
lowest_A2B_d_loss_list_valid_index = np.where(
    A_d_loss_list_valid == lowest_A2B_d_loss_valid)
lowest_B2A_d_loss_list_valid_index = np.where(
    B_d_loss_list_valid == lowest_B2A_d_loss_valid)

# Choosing the epoch with the lowest identity loss valid to plot the result
lowest_A2B_id_loss_valid = np.min(A2A_id_loss_list_valid)
lowest_B2B_id_loss_valid = np.min(B2B_id_loss_list_valid)
lowest_A2B_id_loss_list_valid_index = np.where(
    A2A_id_loss_list_valid == lowest_A2B_id_loss_valid)
lowest_B2B_id_loss_list_valid_index = np.where(
    B2B_id_loss_list_valid == lowest_B2B_id_loss_valid)

# Do the same thing with psnr and ssim, but with the highest value
highest_ssim_A2B = np.max(ssim_A2B_list)
highest_psnr_A2B = np.max(psnr_A2B_list)
highest_ssim_B2A = np.max(ssim_B2A_list)
highest_psnr_B2A = np.max(psnr_B2A_list)
highest_ssim_A2B_index = np.where(ssim_A2B_list == highest_ssim_A2B)
highest_psnr_A2B_index = np.where(psnr_A2B_list == highest_psnr_A2B)
highest_ssim_B2A_index = np.where(ssim_B2A_list == highest_ssim_B2A)
highest_psnr_B2A_index = np.where(psnr_B2A_list == highest_psnr_B2A)


# sample
@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A


@tf.function
def sample_B2A(B):
    B2A = G_B2A(B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return B2A, B2A2B


# run
# save_dir = py.join(args.experiment_dir, args.method,
#                    'samples_testing', 'A2B')
save_dir = py.join(args.experiment_dir, args.method,
                   f'samples_testing_{test_args.loss_method}', 'A2B')
py.mkdir(save_dir)

# Print the best epoch for checking
print(
    f'Lowest A2B adversarial loss valid: {lowest_A2B_adversarial_loss_valid}')
print(
    f'Lowest A2B adversarial loss valid index: {lowest_A2B_adversarial_loss_valid_index[0][0]}')
print(f'Lowest A2B total loss valid: {lowest_A2B_total_loss_valid}')
print(
    f'Lowest A2B total loss valid index: {lowest_A2B_total_loss_valid_index[0][0]}')
print(f'Lowest A2B cycle loss valid: {lowest_A2B_cycle_loss_valid}')
print(
    f'Lowest A2B cycle loss valid index: {lowest_A2B_cycle_loss_valid_index[0][0]}')
print(f'Lowest A2B generator loss valid: {lowest_A2B_g_loss_valid}')
print(
    f'Lowest A2B generator loss valid index: {lowest_A2B_g_loss_list_valid_index[0][0]}')
print(f'Lowest A2B discriminator loss valid: {lowest_A2B_d_loss_valid}')
print(
    f'Lowest A2B discriminator loss valid index: {lowest_A2B_d_loss_list_valid_index[0][0]}')
print(f'Lowest A2B identity loss valid: {lowest_A2B_id_loss_valid}')
print(
    f'Lowest A2B identity loss valid index: {lowest_A2B_id_loss_list_valid_index[0][0]}')
print(f'Highest SSIM A2B: {highest_ssim_A2B}')
print(f'Highest SSIM A2B index: {highest_ssim_A2B_index[0][0]}')
print(f'Highest PSNR A2B: {highest_psnr_A2B}')
print(f'Highest PSNR A2B index: {highest_psnr_A2B_index[0][0]}')
# restore
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
#     args.experiment_dir, args.method, 'checkpoints')).restore()
# Restore the lowest_A2B_adversarial_loss_valid_index
checkDir = 'output/' + method + '/checkpoints'
# use loss method to decide whether to use adversarial or total loss to restore
if test_args.loss_method == 'adversarial':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_adversarial_loss_valid_index[0][0]}')
elif test_args.loss_method == 'total':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_total_loss_valid_index[0][0]}')
elif test_args.loss_method == 'cycle':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_cycle_loss_valid_index[0][0]}')
elif test_args.loss_method == 'generator':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_g_loss_list_valid_index[0][0]}')
elif test_args.loss_method == 'discriminator':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_d_loss_list_valid_index[0][0]}')
elif test_args.loss_method == 'identity':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_id_loss_list_valid_index[0][0]}')
elif test_args.loss_method == 'all':
    # Adversarial cases
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_adversarial_loss_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_adversarial', 'A2B')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
            # psnr, ssim = ev.plot_images_A2B(A_i, A2B_i, B_i,
            #                                 save_dir, A_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_ssim=True)
            i += 1
    # Total case
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_total_loss_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_total', 'A2B')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
            # psnr, ssim = ev.plot_images_A2B(A_i, A2B_i, B_i,
            #                                 save_dir, A_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())

            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_ssim=True)
            i += 1
    # Cycle case
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_cycle_loss_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_cycle', 'A2B')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
            # psnr, ssim = ev.plot_images_A2B(A_i, A2B_i, B_i,
            #                                 save_dir, A_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_ssim=True)

            i += 1

    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_g_loss_list_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_generator', 'A2B')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
            # psnr, ssim = ev.plot_images_A2B(A_i, A2B_i, B_i,
            #                                 save_dir, A_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_ssim=True)
            i += 1

    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_d_loss_list_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_discriminator', 'A2B')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
            # psnr, ssim = ev.plot_images_A2B(A_i, A2B_i, B_i,
            #                                 save_dir, A_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_ssim=True)
            i += 1

    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_A2B_id_loss_list_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_identity', 'A2B')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
            # psnr, ssim = ev.plot_images_A2B(A_i, A2B_i, B_i,
            #                                 save_dir, A_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_ssim=True)

            i += 1

    # Restore the lowest_B2A_adversarial_loss_valid_index
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2A_adversarial_loss_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_adversarial', 'B2A')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        B2A, B2A2B = sample_B2A(B)
        for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
            # psnr, ssim = ev.plot_images_B2A(B_i, B2A_i, A_i,
            #                                 save_dir, B_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_ssim=True)
            i += 1

    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2A_total_loss_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_total', 'B2A')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        B2A, B2A2B = sample_B2A(B)
        for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
            # psnr, ssim = ev.plot_images_B2A(B_i, B2A_i, A_i,
            #                                 save_dir, B_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_ssim=True)
            i += 1

    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2A_cycle_loss_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_cycle', 'B2A')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        B2A, B2A2B = sample_B2A(B)
        for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
            # psnr, ssim = ev.plot_images_B2A(B_i, B2A_i, A_i,
            #                                 save_dir, B_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_ssim=True)
            i += 1

    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2A_g_loss_list_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_generator', 'B2A')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        B2A, B2A2B = sample_B2A(B)
        for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
            # psnr, ssim = ev.plot_images_B2A(B_i, B2A_i, A_i,
            #                                 save_dir, B_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_ssim=True)
            i += 1

    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2A_d_loss_list_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_discriminator', 'B2A')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        B2A, B2A2B = sample_B2A(B)
        for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
            # psnr, ssim = ev.plot_images_B2A(B_i, B2A_i, A_i,
            #                                 save_dir, B_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_ssim=True)
            i += 1

    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2B_id_loss_list_valid_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_identity', 'B2A')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        B2A, B2A2B = sample_B2A(B)
        for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
            # psnr, ssim = ev.plot_images_B2A(B_i, B2A_i, A_i,
            #                                 save_dir, B_img_paths_test[i])
            psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_ssim=True)
            i += 1

    # Exit the code
    sys.exit()
else:
    # Plot the last epoch
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_last', 'A2B')
    py.mkdir(save_dir)

    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore()


i = 0
best_psnr = None
best_ssim = None
for A, B in zip(A_dataset_test, B_dataset_test):
    A2B, A2B2A = sample_A2B(A)
    for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
        # img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B2A_i.numpy()], axis=1)
        # im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
        # img = np.concatenate([A_i.numpy(), A2B_i.numpy(), B_i.numpy()], axis=1)
        # im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))

        # psnr, ssim = ev.plot_images_A2B(A_i, A2B_i, B_i,
        #                                 save_dir, A_img_paths_test[i])
        psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
        if best_psnr is None or psnr > best_psnr:
            best_psnr = psnr
            _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                      save_dir, A_img_paths_test[i], best_psnr=True)
        if best_ssim is None or ssim > best_ssim:
            best_ssim = ssim
            _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                      save_dir, A_img_paths_test[i], best_ssim=True)
        i += 1

# Restore the lowest_B2A_adversarial_loss_valid_index
if test_args.loss_method == 'adversarial':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2A_adversarial_loss_valid_index[0][0]}')
elif test_args.loss_method == 'total':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2A_total_loss_valid_index[0][0]}')
elif test_args.loss_method == 'cycle':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2A_cycle_loss_valid_index[0][0]}')
elif test_args.loss_method == 'generator':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2A_g_loss_list_valid_index[0][0]}')
elif test_args.loss_method == 'discriminator':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2A_d_loss_list_valid_index[0][0]}')
elif test_args.loss_method == 'identity':
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{lowest_B2A_id_loss_list_valid_index[0][0]}')
else:
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore()
# save_dir = py.join(args.experiment_dir, args.method,
#                    'samples_testing', 'B2A')
save_dir = py.join(args.experiment_dir, args.method,
                   f'samples_testing_{test_args.loss_method}', 'B2A')
py.mkdir(save_dir)
i = 0
# Print the best epoch for checking
print(
    f'Lowest B2A adversarial loss valid: {lowest_B2A_adversarial_loss_valid}')
print(
    f'Lowest B2A adversarial loss valid index: {lowest_B2A_adversarial_loss_valid_index[0][0]}')
print(f'Lowest B2A total loss valid: {lowest_B2A_total_loss_valid}')
print(
    f'Lowest B2A total loss valid index: {lowest_B2A_total_loss_valid_index[0][0]}')
for A, B in zip(A_dataset_test, B_dataset_test):
    B2A, B2A2B = sample_B2A(B)
    for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
        # img = np.concatenate([B_i.numpy(), B2A_i.numpy(), B2A2B_i.numpy()], axis=1)
        # im.imwrite(img, py.join(save_dir, py.name_ext(B_img_paths_test[i])))
        # img = np.concatenate([B_i.numpy(), B2A_i.numpy(), A_i.numpy()], axis=1)
        # im.imwrite(img, py.join(save_dir, py.name_ext(B_img_paths_test[i])))
        # psnr, ssim = ev.plot_images_B2A(B_i, B2A_i, A_i,
        #                                 save_dir, B_img_paths_test[i])
        psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
        if best_psnr is None or psnr > best_psnr:
            best_psnr = psnr
            _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                      save_dir, B_img_paths_test[i], best_psnr=True)
        if best_ssim is None or ssim > best_ssim:
            best_ssim = ssim
            _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                      save_dir, B_img_paths_test[i], best_ssim=True)

        i += 1


# Do the same thing with psnr and ssim method
best_psnr = None
best_ssim = None
if test_args.evaluation_method == 'none':
    # Exit the program
    sys.exit()
elif test_args.evaluation_method == 'psnr':
    # Restore the highest_psnr_A2B_index
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{highest_psnr_A2B_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_psnr', 'A2B')
    py.mkdir(save_dir)
    i = 0
    for A, B in zip(A_dataset_test, B_dataset_test):
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
            psnr, ssim = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                            save_dir, A_img_paths_test[i], best_psnr=False)
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_ssim=True)
            i += 1

    # Restore the highest_psnr_B2A_index
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{highest_psnr_B2A_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_psnr', 'B2A')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        B2A, B2A2B = sample_B2A(B)
        for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
            psnr, ssim = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                            save_dir, B_img_paths_test[i], best_psnr=False)
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_ssim=True)
            i += 1
elif test_args.evaluation_method == 'ssim':
    # Restore the highest_ssim_A2B_index
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{highest_ssim_A2B_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_ssim', 'A2B')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
            psnr, ssim = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                            save_dir, A_img_paths_test[i])
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_ssim=True)
            i += 1

    # Restore the highest_ssim_B2A_index
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{highest_ssim_B2A_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_ssim', 'B2A')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        B2A, B2A2B = sample_B2A(B)
        for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
            psnr, ssim = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                            save_dir, B_img_paths_test[i])
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_ssim=True)
            i += 1
elif test_args.evaluation_method == 'both':
    # Restore the highest_psnr_A2B_index
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{highest_psnr_A2B_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_psnr', 'A2B')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
            psnr, ssim = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                            save_dir, A_img_paths_test[i], best_psnr=False)
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_ssim=True)
            i += 1

    # Restore the highest_ssim_A2B_index
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{highest_ssim_A2B_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_ssim', 'A2B')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
            psnr, ssim = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                            save_dir, A_img_paths_test[i])
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_A2B(A_i, A2B_i, B_i,
                                          save_dir, A_img_paths_test[i], best_ssim=True)
            i += 1

    # Restore the highest_psnr_B2A_index
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(checkDir +
                                                                  f'/ckpt-{highest_psnr_B2A_index[0][0]}')
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_psnr', 'B2A')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        B2A, B2A2B = sample_B2A(B)
        for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
            psnr, ssim = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                            save_dir, B_img_paths_test[i], best_psnr=False)
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_ssim=True)
            i += 1

    # Restore the highest_ssim_B2A_index
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
        args.experiment_dir, args.method, 'checkpoints')).restore(
            checkDir + f'/ckpt-{highest_ssim_B2A_index[0][0]}'
    )
    save_dir = py.join(args.experiment_dir, args.method,
                       f'samples_testing_ssim', 'B2A')
    py.mkdir(save_dir)
    i = 0
    best_psnr = None
    best_ssim = None
    for A, B in zip(A_dataset_test, B_dataset_test):
        B2A, B2A2B = sample_B2A(B)
        for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
            psnr, ssim = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                            save_dir, B_img_paths_test[i])
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_psnr=True)
            if best_ssim is None or ssim > best_ssim:
                best_ssim = ssim
                _, _ = ev.plot_images_B2A(B_i, B2A_i, A_i,
                                          save_dir, B_img_paths_test[i], best_ssim=True)
            i += 1
