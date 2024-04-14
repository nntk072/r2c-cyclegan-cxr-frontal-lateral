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
py.arg('--method', help='convolutional, operational, unet, anotherunet, operational_unet, all',
       default='all')
py.arg('--loss_method', help='none, adversarial, total, cycle, generator, discriminator, identity, all',
       default='all')


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = py.glob(
    py.join('datasets', 'testA'), '*.jpg')
B_img_paths_test = py.glob(
    py.join('datasets', 'testB'), '*.jpg')
A_dataset_test = data.make_dataset(A_img_paths_test, 1, 286, 256,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
B_dataset_test = data.make_dataset(B_img_paths_test, 1, 286, 256,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
test_args = py.args()


def get_params(test_args, method):
    args = py.args_from_yaml(
        py.join(test_args.experiment_dir, method, 'settings.yml'))
    # args.__dict__.update(test_args.__dict__)
    return args


def get_values(method, args, epochs):
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
    for ep in range(0, epochs):
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
    for ep in range(0, epochs):  # the name of the folder is validation
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
    A2B_adversarial_loss_valid = np.add(
        A2B_g_loss_list_valid, A_d_loss_list_valid)
    B2A_adversarial_loss_valid = np.add(
        B2A_g_loss_list_valid, B_d_loss_list_valid)

    # Calculate the total loss, remember tot ake the cycle_loss_weight from args
    A2B_total_loss = np.add(A2B_adversarial_loss,
                            np.multiply(A2B2A_cycle_loss_list, args.cycle_loss_weight))
    B2A_total_loss = np.add(B2A_adversarial_loss,
                            np.multiply(B2A2B_cycle_loss_list, args.cycle_loss_weight))
    A2B_total_loss_valid = np.add(A2B_adversarial_loss_valid,
                                  np.multiply(A2B2A_cycle_loss_list_valid, args.cycle_loss_weight))
    B2A_total_loss_valid = np.add(B2A_adversarial_loss_valid,
                                  np.multiply(B2A2B_cycle_loss_list_valid, args.cycle_loss_weight))

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

    # return a dictionary with lowest and highest values
    return {'lowest_A2B_adversarial_loss_valid': lowest_A2B_adversarial_loss_valid,
            'lowest_B2A_adversarial_loss_valid': lowest_B2A_adversarial_loss_valid,
            'lowest_A2B_adversarial_loss_valid_index': lowest_A2B_adversarial_loss_valid_index,
            'lowest_B2A_adversarial_loss_valid_index': lowest_B2A_adversarial_loss_valid_index,
            'lowest_A2B_total_loss_valid': lowest_A2B_total_loss_valid,
            'lowest_B2A_total_loss_valid': lowest_B2A_total_loss_valid,
            'lowest_A2B_total_loss_valid_index': lowest_A2B_total_loss_valid_index,
            'lowest_B2A_total_loss_valid_index': lowest_B2A_total_loss_valid_index,
            'lowest_A2B_cycle_loss_valid': lowest_A2B_cycle_loss_valid,
            'lowest_B2A_cycle_loss_valid': lowest_B2A_cycle_loss_valid,
            'lowest_A2B_cycle_loss_valid_index': lowest_A2B_cycle_loss_valid_index,
            'lowest_B2A_cycle_loss_valid_index': lowest_B2A_cycle_loss_valid_index,
            'lowest_A2B_g_loss_valid': lowest_A2B_g_loss_valid,
            'lowest_B2A_g_loss_valid': lowest_B2A_g_loss_valid,
            'lowest_A2B_g_loss_list_valid_index': lowest_A2B_g_loss_list_valid_index,
            'lowest_B2A_g_loss_list_valid_index': lowest_B2A_g_loss_list_valid_index,
            'lowest_A2B_d_loss_valid': lowest_A2B_d_loss_valid,
            'lowest_B2A_d_loss_valid': lowest_B2A_d_loss_valid,
            'lowest_A2B_d_loss_list_valid_index': lowest_A2B_d_loss_list_valid_index,
            'lowest_B2A_d_loss_list_valid_index': lowest_B2A_d_loss_list_valid_index,
            'lowest_A2B_id_loss_valid': lowest_A2B_id_loss_valid,
            'lowest_B2B_id_loss_valid': lowest_B2B_id_loss_valid,
            'lowest_A2B_id_loss_list_valid_index': lowest_A2B_id_loss_list_valid_index,
            'lowest_B2B_id_loss_list_valid_index': lowest_B2B_id_loss_list_valid_index,
            'highest_ssim_A2B': highest_ssim_A2B,
            'highest_psnr_A2B': highest_psnr_A2B,
            'highest_ssim_B2A': highest_ssim_B2A,
            'highest_psnr_B2A': highest_psnr_B2A,
            'highest_ssim_A2B_index': highest_ssim_A2B_index,
            'highest_psnr_A2B_index': highest_psnr_A2B_index,
            'highest_ssim_B2A_index': highest_ssim_B2A_index,
            'highest_psnr_B2A_index': highest_psnr_B2A_index,
            }


# Do the same methods, but with all methods
method = ["convolutional", "operational", "unet", "operational_unet"]

# Get the parameter for each method
convolutional_args = None
operational_args = None
unet_args = None
operational_unet_args = None
convolutional_values = None
operational_values = None
unet_values = None
operational_unet_values = None
for m in method:
    if m == "convolutional":
        convolutional_args = get_params(test_args, m)
        convolutional_values = get_values(m, convolutional_args, 1000)
    elif m == "operational":
        operational_args = get_params(test_args, m)
        operational_values = get_values(m, operational_args, 1000)
    elif m == "unet":
        unet_args = get_params(test_args, m)
        unet_values = get_values(m, unet_args, 1000)
    elif m == "operational_unet":
        operational_unet_args = get_params(test_args, m)
        operational_unet_values = get_values(m, operational_unet_args, 1000)

output = []
# Instead of appending manually, just loop through the dictionary and append
for key, value in convolutional_values.items():
    output.append(f'Convolutional {key}: {value}')
for key, value in operational_values.items():
    output.append(f'Operational {key}: {value}')
for key, value in unet_values.items():
    output.append(f'UNet {key}: {value}')
for key, value in operational_unet_values.items():
    output.append(f'Operational UNet {key}: {value}')


# Save the output into a txt file
output_file = open(
    f'output/{test_args.method}/output.txt', 'w')
for line in output:
    output_file.write(line + '\n')
output_file.close()


G_A2B = module.ResnetGenerator(
    input_shape=(convolutional_args.crop_size, convolutional_args.crop_size, 3))
G_B2A = module.ResnetGenerator(
    input_shape=(convolutional_args.crop_size, convolutional_args.crop_size, 3))
G_A2B_op = module.OpGenerator(input_shape=(
    operational_args.crop_size, operational_args.crop_size, 3), q=operational_args.q)
G_B2A_op = module.OpGenerator(input_shape=(
    operational_args.crop_size, operational_args.crop_size, 3), q=operational_args.q)
G_A2B_unet = module.UNetGenerator(
    input_shape=(unet_args.crop_size, unet_args.crop_size, 3))
G_B2A_unet = module.UNetGenerator(
    input_shape=(unet_args.crop_size, unet_args.crop_size, 3))
# G_A2B_opunet = module.OpUNetGenerator(
#     input_shape=(operational_unet_args.crop_size, operational_unet_args.crop_size, 3), q=operational_unet_args.q)
# G_B2A_opunet = module.OpUNetGenerator(
#     input_shape=(operational_unet_args.crop_size, operational_unet_args.crop_size, 3), q=operational_unet_args.q)
G_A2B_opunet = module.UNetGenerator(
    input_shape=(unet_args.crop_size, unet_args.crop_size, 3))
G_B2A_opunet = module.UNetGenerator(
    input_shape=(unet_args.crop_size, unet_args.crop_size, 3))

# checkDir = 'output/' + method + '/checkpoints'
checkDir = 'output/' + 'convolutional/' + 'checkpoints'
checkDir_op = 'output/' + 'operational/' + 'checkpoints'
checkDir_unet = 'output/' + 'unet/' + 'checkpoints'
checkDir_opunet = 'output/' + 'operational_unet/' + 'checkpoints'

# sample


@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    A2B_op = G_A2B_op(A, training=False)
    A2B2A_op = G_B2A_op(A2B_op, training=False)
    A2B_unet = G_A2B_unet(A, training=False)
    A2B2A_unet = G_B2A_unet(A2B_unet, training=False)
    A2B_opunet = G_A2B_opunet(A, training=False)
    A2B2A_opunet = G_B2A_opunet(A2B_opunet, training=False)
    return A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet


@tf.function
def sample_B2A(B):
    B2A = G_B2A(B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    B2A_op = G_B2A_op(B, training=False)
    B2A2B_op = G_A2B_op(B2A_op, training=False)
    B2A_unet = G_B2A_unet(B, training=False)
    B2A2B_unet = G_A2B_unet(B2A_unet, training=False)
    B2A_opunet = G_B2A_opunet(B, training=False)
    B2A2B_opunet = G_A2B_opunet(B2A_opunet, training=False)
    return B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet


# # ADVERSARIAL LOSSES
# # Restore all the checkpoints from all methods
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
#     test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
#                                                                        f'/ckpt-{convolutional_values["lowest_A2B_adversarial_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
#     test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
#                                                                      f'/ckpt-{operational_values["lowest_A2B_adversarial_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
#     test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
#                                                               f'/ckpt-{unet_values["lowest_A2B_adversarial_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
#     test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet + f'/ckpt-{operational_unet_values["lowest_A2B_adversarial_loss_valid_index"][0][0]}')

# # Set the save directory for the samples
# save_dir = py.join(test_args.experiment_dir, test_args.method,
#                    f'samples_testing_adversarial', 'A2B')
# py.mkdir(save_dir)
# i = 0

# for A, B in zip(A_dataset_test, B_dataset_test):
#     A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet = sample_A2B(
#         A)
#     for A_i, A2B_i, A2B2A_i, A2B_op_i, A2B2A_op_i, A2B_unet_i, A2B2A_unet_i, A2B_opunet_i, A2B2A_opunet_i, B_i in zip(A, A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet, B):
#         psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
#         psnr_op, ssim_op = ev.compute_psnr_ssim(A2B_op_i.numpy(), B_i.numpy())
#         psnr_unet, ssim_unet = ev.compute_psnr_ssim(
#             A2B_unet_i.numpy(), B_i.numpy())
#         psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
#             A2B_opunet_i.numpy(), B_i.numpy())
#         A2B_list = [A2B_i, A2B_op_i, A2B_unet_i, A2B_opunet_i]
#         A2B2A_list = [A2B2A_i, A2B2A_op_i, A2B2A_unet_i, A2B2A_opunet_i]
#         psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
#         ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]
#         method_list = ["convolutional",
#                        "operational", "unet", "operational_unet"]
#         ev.plot_all_images_A2B(A_i, A2B_list, B_i, psnr_list,
#                                ssim_list, save_dir, A_img_paths_test[i], method_list)
#         i += 1
# # Restore the checkpoints from all methods
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
#     test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
#                                                                        f'/ckpt-{convolutional_values["lowest_B2A_adversarial_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
#     test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
#                                                                      f'/ckpt-{operational_values["lowest_B2A_adversarial_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
#     test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
#                                                               f'/ckpt-{unet_values["lowest_B2A_adversarial_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
#     test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
#                                                                           f'/ckpt-{operational_unet_values["lowest_B2A_adversarial_loss_valid_index"][0][0]}')

# # Set the save directory for the samples
# save_dir = py.join(test_args.experiment_dir, test_args.method,
#                    f'samples_testing_adversarial', 'B2A')
# py.mkdir(save_dir)
# i = 0
# for A, B in zip(A_dataset_test, B_dataset_test):
#     B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet = sample_B2A(
#         B)
#     for B_i, B2A_i, B2A2B_i, B2A_op_i, B2A2B_op_i, B2A_unet_i, B2A2B_unet_i, B2A_opunet_i, B2A2B_opunet_i, A_i in zip(B, B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet, A):
#         psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
#         psnr_op, ssim_op = ev.compute_psnr_ssim(B2A_op_i.numpy(), A_i.numpy())
#         psnr_unet, ssim_unet = ev.compute_psnr_ssim(
#             B2A_unet_i.numpy(), A_i.numpy())
#         psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
#             B2A_opunet_i.numpy(), A_i.numpy())
#         B2A_list = [B2A_i, B2A_op_i, B2A_unet_i, B2A_opunet_i]
#         B2A2B_list = [B2A2B_i, B2A2B_op_i, B2A2B_unet_i, B2A2B_opunet_i]
#         psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
#         ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]

#         method_list = ["convolutional",
#                        "operational", "unet", "operational_unet"]
#         ev.plot_all_images_B2A(B_i, B2A_list, A_i, psnr_list,
#                                ssim_list, save_dir, A_img_paths_test[i], method_list)
#         i += 1

# # TOTAL LOSSES
# # Restore all the checkpoints from all methods
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
#     test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
#                                                                        f'/ckpt-{convolutional_values["lowest_A2B_total_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
#     test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
#                                                                      f'/ckpt-{operational_values["lowest_A2B_total_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
#     test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
#                                                               f'/ckpt-{unet_values["lowest_A2B_total_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
#     test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
#                                                                           f'/ckpt-{operational_unet_values["lowest_A2B_total_loss_valid_index"][0][0]}')

# # Set the save directory for the samples
# save_dir = py.join(test_args.experiment_dir, test_args.method,
#                    f'samples_testing_total', 'A2B')
# py.mkdir(save_dir)
# i = 0

# for A, B in zip(A_dataset_test, B_dataset_test):
#     A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet = sample_A2B(
#         A)
#     for A_i, A2B_i, A2B2A_i, A2B_op_i, A2B2A_op_i, A2B_unet_i, A2B2A_unet_i, A2B_opunet_i, A2B2A_opunet_i, B_i in zip(A, A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet, B):
#         psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
#         psnr_op, ssim_op = ev.compute_psnr_ssim(A2B_op_i.numpy(), B_i.numpy())
#         psnr_unet, ssim_unet = ev.compute_psnr_ssim(
#             A2B_unet_i.numpy(), B_i.numpy())
#         psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
#             A2B_opunet_i.numpy(), B_i.numpy())
#         A2B_list = [A2B_i, A2B_op_i, A2B_unet_i, A2B_opunet_i]
#         A2B2A_list = [A2B2A_i, A2B2A_op_i, A2B2A_unet_i, A2B2A_opunet_i]
#         psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
#         ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]
#         method_list = ["convolutional",
#                        "operational", "unet", "operational_unet"]
#         ev.plot_all_images_A2B(A_i, A2B_list, B_i, psnr_list,
#                                ssim_list, save_dir, A_img_paths_test[i], method_list)
#         i += 1

# # Restore the checkpoints from all methods
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
#     test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
#                                                                        f'/ckpt-{convolutional_values["lowest_B2A_total_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
#     test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
#                                                                      f'/ckpt-{operational_values["lowest_B2A_total_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
#     test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
#                                                               f'/ckpt-{unet_values["lowest_B2A_total_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
#     test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
#                                                                           f'/ckpt-{operational_unet_values["lowest_B2A_total_loss_valid_index"][0][0]}')

# # Set the save directory for the samples
# save_dir = py.join(test_args.experiment_dir, test_args.method,
#                    f'samples_testing_total', 'B2A')
# py.mkdir(save_dir)
# i = 0
# for A, B in zip(A_dataset_test, B_dataset_test):
#     B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet = sample_B2A(
#         B)
#     for B_i, B2A_i, B2A2B_i, B2A_op_i, B2A2B_op_i, B2A_unet_i, B2A2B_unet_i, B2A_opunet_i, B2A2B_opunet_i, A_i in zip(B, B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet, A):
#         psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
#         psnr_op, ssim_op = ev.compute_psnr_ssim(B2A_op_i.numpy(), A_i.numpy())
#         psnr_unet, ssim_unet = ev.compute_psnr_ssim(
#             B2A_unet_i.numpy(), A_i.numpy())
#         psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
#             B2A_opunet_i.numpy(), A_i.numpy())
#         B2A_list = [B2A_i, B2A_op_i, B2A_unet_i, B2A_opunet_i]
#         B2A2B_list = [B2A2B_i, B2A2B_op_i, B2A2B_unet_i, B2A2B_opunet_i]
#         psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
#         ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]

#         method_list = ["convolutional",
#                        "operational", "unet", "operational_unet"]
#         ev.plot_all_images_B2A(B_i, B2A_list, A_i, psnr_list,
#                                ssim_list, save_dir, A_img_paths_test[i], method_list)
#         i += 1

# # CYCLE LOSSES
# # Restore all the checkpoints from all methods
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
#     test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
#                                                                        f'/ckpt-{convolutional_values["lowest_A2B_cycle_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
#     test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
#                                                                      f'/ckpt-{operational_values["lowest_A2B_cycle_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
#     test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
#                                                               f'/ckpt-{unet_values["lowest_A2B_cycle_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
#     test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
#                                                                           f'/ckpt-{operational_unet_values["lowest_A2B_cycle_loss_valid_index"][0][0]}')

# # Set the save directory for the samples
# save_dir = py.join(test_args.experiment_dir, test_args.method,
#                    f'samples_testing_cycle', 'A2B')
# py.mkdir(save_dir)
# i = 0

# for A, B in zip(A_dataset_test, B_dataset_test):
#     A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet = sample_A2B(
#         A)
#     for A_i, A2B_i, A2B2A_i, A2B_op_i, A2B2A_op_i, A2B_unet_i, A2B2A_unet_i, A2B_opunet_i, A2B2A_opunet_i, B_i in zip(A, A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet, B):
#         psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
#         psnr_op, ssim_op = ev.compute_psnr_ssim(A2B_op_i.numpy(), B_i.numpy())
#         psnr_unet, ssim_unet = ev.compute_psnr_ssim(
#             A2B_unet_i.numpy(), B_i.numpy())
#         psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
#             A2B_opunet_i.numpy(), B_i.numpy())
#         A2B_list = [A2B_i, A2B_op_i, A2B_unet_i, A2B_opunet_i]
#         A2B2A_list = [A2B2A_i, A2B2A_op_i, A2B2A_unet_i, A2B2A_opunet_i]
#         psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
#         ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]
#         method_list = ["convolutional",
#                        "operational", "unet", "operational_unet"]
#         ev.plot_all_images_A2B(A_i, A2B_list, B_i, psnr_list,
#                                ssim_list, save_dir, A_img_paths_test[i], method_list)
#         i += 1

# # Restore the checkpoints from all methods
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
#     test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
#                                                                        f'/ckpt-{convolutional_values["lowest_B2A_cycle_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
#     test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
#                                                                      f'/ckpt-{operational_values["lowest_B2A_cycle_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
#     test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
#                                                               f'/ckpt-{unet_values["lowest_B2A_cycle_loss_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
#     test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
#                                                                           f'/ckpt-{operational_unet_values["lowest_B2A_cycle_loss_valid_index"][0][0]}')

# # Set the save directory for the samples
# save_dir = py.join(test_args.experiment_dir, test_args.method,
#                    f'samples_testing_cycle', 'B2A')
# py.mkdir(save_dir)
# i = 0
# for A, B in zip(A_dataset_test, B_dataset_test):
#     B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet = sample_B2A(
#         B)
#     for B_i, B2A_i, B2A2B_i, B2A_op_i, B2A2B_op_i, B2A_unet_i, B2A2B_unet_i, B2A_opunet_i, B2A2B_opunet_i, A_i in zip(B, B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet, A):
#         psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
#         psnr_op, ssim_op = ev.compute_psnr_ssim(B2A_op_i.numpy(), A_i.numpy())
#         psnr_unet, ssim_unet = ev.compute_psnr_ssim(
#             B2A_unet_i.numpy(), A_i.numpy())
#         psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
#             B2A_opunet_i.numpy(), A_i.numpy())
#         B2A_list = [B2A_i, B2A_op_i, B2A_unet_i, B2A_opunet_i]
#         B2A2B_list = [B2A2B_i, B2A2B_op_i, B2A2B_unet_i, B2A2B_opunet_i]
#         psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
#         ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]

#         method_list = ["convolutional",
#                        "operational", "unet", "operational_unet"]
#         ev.plot_all_images_B2A(B_i, B2A_list, A_i, psnr_list,
#                                ssim_list, save_dir, A_img_paths_test[i], method_list)
#         i += 1

# # GENERATOR LOSSES
# # Restore all the checkpoints from all methods
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
#     test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
#                                                                        f'/ckpt-{convolutional_values["lowest_A2B_g_loss_list_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
#     test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
#                                                                      f'/ckpt-{operational_values["lowest_A2B_g_loss_list_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
#     test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
#                                                               f'/ckpt-{unet_values["lowest_A2B_g_loss_list_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
#     test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
#                                                                           f'/ckpt-{operational_unet_values["lowest_A2B_g_loss_list_valid_index"][0][0]}')

# # Set the save directory for the samples
# save_dir = py.join(test_args.experiment_dir, test_args.method,
#                    f'samples_testing_generator', 'A2B')
# py.mkdir(save_dir)
# i = 0

# for A, B in zip(A_dataset_test, B_dataset_test):
#     A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet = sample_A2B(
#         A)
#     for A_i, A2B_i, A2B2A_i, A2B_op_i, A2B2A_op_i, A2B_unet_i, A2B2A_unet_i, A2B_opunet_i, A2B2A_opunet_i, B_i in zip(A, A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet, B):
#         psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
#         psnr_op, ssim_op = ev.compute_psnr_ssim(A2B_op_i.numpy(), B_i.numpy())
#         psnr_unet, ssim_unet = ev.compute_psnr_ssim(
#             A2B_unet_i.numpy(), B_i.numpy())
#         psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
#             A2B_opunet_i.numpy(), B_i.numpy())
#         A2B_list = [A2B_i, A2B_op_i, A2B_unet_i, A2B_opunet_i]
#         A2B2A_list = [A2B2A_i, A2B2A_op_i, A2B2A_unet_i, A2B2A_opunet_i]
#         psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
#         ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]
#         method_list = ["convolutional",
#                        "operational", "unet", "operational_unet"]
#         ev.plot_all_images_A2B(A_i, A2B_list, B_i, psnr_list,
#                                ssim_list, save_dir, A_img_paths_test[i], method_list)
#         i += 1

# # Restore the checkpoints from all methods
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
#     test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
#                                                                        f'/ckpt-{convolutional_values["lowest_B2A_g_loss_list_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
#     test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
#                                                                      f'/ckpt-{operational_values["lowest_B2A_g_loss_list_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
#     test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
#                                                               f'/ckpt-{unet_values["lowest_B2A_g_loss_list_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
#     test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
#                                                                           f'/ckpt-{operational_unet_values["lowest_B2A_g_loss_list_valid_index"][0][0]}')

# # Set the save directory for the samples
# save_dir = py.join(test_args.experiment_dir, test_args.method,
#                    f'samples_testing_generator', 'B2A')
# py.mkdir(save_dir)
# i = 0
# for A, B in zip(A_dataset_test, B_dataset_test):
#     B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet = sample_B2A(
#         B)
#     for B_i, B2A_i, B2A2B_i, B2A_op_i, B2A2B_op_i, B2A_unet_i, B2A2B_unet_i, B2A_opunet_i, B2A2B_opunet_i, A_i in zip(B, B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet, A):
#         psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
#         psnr_op, ssim_op = ev.compute_psnr_ssim(B2A_op_i.numpy(), A_i.numpy())
#         psnr_unet, ssim_unet = ev.compute_psnr_ssim(
#             B2A_unet_i.numpy(), A_i.numpy())
#         psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
#             B2A_opunet_i.numpy(), A_i.numpy())
#         B2A_list = [B2A_i, B2A_op_i, B2A_unet_i, B2A_opunet_i]
#         B2A2B_list = [B2A2B_i, B2A2B_op_i, B2A2B_unet_i, B2A2B_opunet_i]
#         psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
#         ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]

#         method_list = ["convolutional",
#                        "operational", "unet", "operational_unet"]
#         ev.plot_all_images_B2A(B_i, B2A_list, A_i, psnr_list,
#                                ssim_list, save_dir, A_img_paths_test[i], method_list)
#         i += 1

# # IDENTITY LOSSES
# # Restore all the checkpoints from all methods
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
#     test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
#                                                                        f'/ckpt-{convolutional_values["lowest_A2B_d_loss_list_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
#     test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
#                                                                      f'/ckpt-{operational_values["lowest_A2B_d_loss_list_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
#     test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
#                                                               f'/ckpt-{unet_values["lowest_A2B_d_loss_list_valid_index"][0][0]}')
# tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
#     test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
#                                                                           f'/ckpt-{operational_unet_values["lowest_A2B_d_loss_list_valid_index"][0][0]}')

# # Set the save directory for the samples
# save_dir = py.join(test_args.experiment_dir, test_args.method,
#                    f'samples_testing_identity', 'A2B')
# py.mkdir(save_dir)
# i = 0

# for A, B in zip(A_dataset_test, B_dataset_test):
#     A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet = sample_A2B(
#         A)
#     for A_i, A2B_i, A2B2A_i, A2B_op_i, A2B2A_op_i, A2B_unet_i, A2B2A_unet_i, A2B_opunet_i, A2B2A_opunet_i, B_i in zip(A, A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet, B):
#         psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
#         psnr_op, ssim_op = ev.compute_psnr_ssim(A2B_op_i.numpy(), B_i.numpy())
#         psnr_unet, ssim_unet = ev.compute_psnr_ssim(
#             A2B_unet_i.numpy(), B_i.numpy())
#         psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
#             A2B_opunet_i.numpy(), B_i.numpy())
#         A2B_list = [A2B_i, A2B_op_i, A2B_unet_i, A2B_opunet_i]
#         A2B2A_list = [A2B2A_i, A2B2A_op_i, A2B2A_unet_i, A2B2A_opunet_i]
#         psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
#         ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]
#         method_list = ["convolutional",
#                        "operational", "unet", "operational_unet"]
#         ev.plot_all_images_A2B(A_i, A2B_list, B_i, psnr_list,
#                                ssim_list, save_dir, A_img_paths_test[i], method_list)
#         i += 1

# Restore the checkpoints from all methods
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
    test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
                                                                       f'/ckpt-{convolutional_values["lowest_B2A_d_loss_list_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
    test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
                                                                     f'/ckpt-{operational_values["lowest_B2A_d_loss_list_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
    test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
                                                              f'/ckpt-{unet_values["lowest_B2A_d_loss_list_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
    test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
                                                                          f'/ckpt-{operational_unet_values["lowest_B2A_d_loss_list_valid_index"][0][0]}')

# Set the save directory for the samples
save_dir = py.join(test_args.experiment_dir, test_args.method,
                   f'samples_testing_identity', 'B2A')
py.mkdir(save_dir)
i = 0
for A, B in zip(A_dataset_test, B_dataset_test):
    B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet = sample_B2A(
        B)
    for B_i, B2A_i, B2A2B_i, B2A_op_i, B2A2B_op_i, B2A_unet_i, B2A2B_unet_i, B2A_opunet_i, B2A2B_opunet_i, A_i in zip(B, B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet, A):
        psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
        psnr_op, ssim_op = ev.compute_psnr_ssim(B2A_op_i.numpy(), A_i.numpy())
        psnr_unet, ssim_unet = ev.compute_psnr_ssim(
            B2A_unet_i.numpy(), A_i.numpy())
        psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
            B2A_opunet_i.numpy(), A_i.numpy())
        B2A_list = [B2A_i, B2A_op_i, B2A_unet_i, B2A_opunet_i]
        B2A2B_list = [B2A2B_i, B2A2B_op_i, B2A2B_unet_i, B2A2B_opunet_i]
        psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
        ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]

        method_list = ["convolutional",
                       "operational", "unet", "operational_unet"]
        ev.plot_all_images_B2A(B_i, B2A_list, A_i, psnr_list,
                               ssim_list, save_dir, A_img_paths_test[i], method_list)
        i += 1

# PSNR
# Restore all the checkpoints from all methods
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
    test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
                                                                       f'/ckpt-{convolutional_values["highest_A2B_psnr_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
    test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
                                                                     f'/ckpt-{operational_values["highest_A2B_psnr_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
    test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
                                                              f'/ckpt-{unet_values["highest_A2B_psnr_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
    test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
                                                                          f'/ckpt-{operational_unet_values["highest_A2B_psnr_valid_index"][0][0]}')

# Set the save directory for the samples
save_dir = py.join(test_args.experiment_dir, test_args.method,
                   f'samples_testing_psnr', 'A2B')
py.mkdir(save_dir)
i = 0

for A, B in zip(A_dataset_test, B_dataset_test):
    A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet = sample_A2B(
        A)
    for A_i, A2B_i, A2B2A_i, A2B_op_i, A2B2A_op_i, A2B_unet_i, A2B2A_unet_i, A2B_opunet_i, A2B2A_opunet_i, B_i in zip(A, A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet, B):
        psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
        psnr_op, ssim_op = ev.compute_psnr_ssim(A2B_op_i.numpy(), B_i.numpy())
        psnr_unet, ssim_unet = ev.compute_psnr_ssim(
            A2B_unet_i.numpy(), B_i.numpy())
        psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
            A2B_opunet_i.numpy(), B_i.numpy())
        A2B_list = [A2B_i, A2B_op_i, A2B_unet_i, A2B_opunet_i]
        A2B2A_list = [A2B2A_i, A2B2A_op_i, A2B2A_unet_i, A2B2A_opunet_i]
        psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
        ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]
        method_list = ["convolutional",
                       "operational", "unet", "operational_unet"]
        ev.plot_all_images_A2B(A_i, A2B_list, B_i, psnr_list,
                               ssim_list, save_dir, A_img_paths_test[i], method_list)
        i += 1

# Restore the checkpoints from all methods
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
    test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
                                                                       f'/ckpt-{convolutional_values["highest_B2A_psnr_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
    test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
                                                                     f'/ckpt-{operational_values["highest_B2A_psnr_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
    test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
                                                              f'/ckpt-{unet_values["highest_B2A_psnr_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
    test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
                                                                          f'/ckpt-{operational_unet_values["highest_B2A_psnr_valid_index"][0][0]}')

# Set the save directory for the samples
save_dir = py.join(test_args.experiment_dir, test_args.method,
                   f'samples_testing_psnr', 'B2A')
py.mkdir(save_dir)
i = 0
for A, B in zip(A_dataset_test, B_dataset_test):
    B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet = sample_B2A(
        B)
    for B_i, B2A_i, B2A2B_i, B2A_op_i, B2A2B_op_i, B2A_unet_i, B2A2B_unet_i, B2A_opunet_i, B2A2B_opunet_i, A_i in zip(B, B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet, A):
        psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
        psnr_op, ssim_op = ev.compute_psnr_ssim(B2A_op_i.numpy(), A_i.numpy())
        psnr_unet, ssim_unet = ev.compute_psnr_ssim(
            B2A_unet_i.numpy(), A_i.numpy())
        psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
            B2A_opunet_i.numpy(), A_i.numpy())
        B2A_list = [B2A_i, B2A_op_i, B2A_unet_i, B2A_opunet_i]
        B2A2B_list = [B2A2B_i, B2A2B_op_i, B2A2B_unet_i, B2A2B_opunet_i]
        psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
        ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]

        method_list = ["convolutional",
                       "operational", "unet", "operational_unet"]
        ev.plot_all_images_B2A(B_i, B2A_list, A_i, psnr_list,
                               ssim_list, save_dir, A_img_paths_test[i], method_list)
        i += 1

# SSIM
# Restore all the checkpoints from all methods
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
    test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
                                                                       f'/ckpt-{convolutional_values["highest_A2B_ssim_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
    test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
                                                                     f'/ckpt-{operational_values["highest_A2B_ssim_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
    test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
                                                              f'/ckpt-{unet_values["highest_A2B_ssim_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
    test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
                                                                          f'/ckpt-{operational_unet_values["highest_A2B_ssim_valid_index"][0][0]}')

# Set the save directory for the samples
save_dir = py.join(test_args.experiment_dir, test_args.method,
                   f'samples_testing_ssim', 'A2B')
py.mkdir(save_dir)
i = 0

for A, B in zip(A_dataset_test, B_dataset_test):
    A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet = sample_A2B(
        A)
    for A_i, A2B_i, A2B2A_i, A2B_op_i, A2B2A_op_i, A2B_unet_i, A2B2A_unet_i, A2B_opunet_i, A2B2A_opunet_i, B_i in zip(A, A2B, A2B2A, A2B_op, A2B2A_op, A2B_unet, A2B2A_unet, A2B_opunet, A2B2A_opunet, B):
        psnr, ssim = ev.compute_psnr_ssim(A2B_i.numpy(), B_i.numpy())
        psnr_op, ssim_op = ev.compute_psnr_ssim(A2B_op_i.numpy(), B_i.numpy())
        psnr_unet, ssim_unet = ev.compute_psnr_ssim(
            A2B_unet_i.numpy(), B_i.numpy())
        psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
            A2B_opunet_i.numpy(), B_i.numpy())
        A2B_list = [A2B_i, A2B_op_i, A2B_unet_i, A2B_opunet_i]
        A2B2A_list = [A2B2A_i, A2B2A_op_i, A2B2A_unet_i, A2B2A_opunet_i]
        psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
        ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]
        method_list = ["convolutional",
                       "operational", "unet", "operational_unet"]
        ev.plot_all_images_A2B(A_i, A2B_list, B_i, psnr_list,
                               ssim_list, save_dir, A_img_paths_test[i], method_list)
        i += 1

# Restore the checkpoints from all methods
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
    test_args.experiment_dir, 'convolutional', 'checkpoints')).restore(checkDir +
                                                                       f'/ckpt-{convolutional_values["highest_B2A_ssim_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_op, G_B2A=G_B2A_op), py.join(
    test_args.experiment_dir, 'operational', 'checkpoints')).restore(checkDir_op +
                                                                     f'/ckpt-{operational_values["highest_B2A_ssim_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_unet, G_B2A=G_B2A_unet), py.join(
    test_args.experiment_dir, 'unet', 'checkpoints')).restore(checkDir_unet +
                                                              f'/ckpt-{unet_values["highest_B2A_ssim_valid_index"][0][0]}')
tl.Checkpoint(dict(G_A2B=G_A2B_opunet, G_B2A=G_B2A_opunet), py.join(
    test_args.experiment_dir, 'operational_unet', 'checkpoints')).restore(checkDir_opunet +
                                                                          f'/ckpt-{operational_unet_values["highest_B2A_ssim_valid_index"][0][0]}')

# Set the save directory for the samples
save_dir = py.join(test_args.experiment_dir, test_args.method,
                   f'samples_testing_ssim', 'B2A')
py.mkdir(save_dir)
i = 0
for A, B in zip(A_dataset_test, B_dataset_test):
    B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet = sample_B2A(
        B)
    for B_i, B2A_i, B2A2B_i, B2A_op_i, B2A2B_op_i, B2A_unet_i, B2A2B_unet_i, B2A_opunet_i, B2A2B_opunet_i, A_i in zip(B, B2A, B2A2B, B2A_op, B2A2B_op, B2A_unet, B2A2B_unet, B2A_opunet, B2A2B_opunet, A):
        psnr, ssim = ev.compute_psnr_ssim(B2A_i.numpy(), A_i.numpy())
        psnr_op, ssim_op = ev.compute_psnr_ssim(B2A_op_i.numpy(), A_i.numpy())
        psnr_unet, ssim_unet = ev.compute_psnr_ssim(
            B2A_unet_i.numpy(), A_i.numpy())
        psnr_opunet, ssim_opunet = ev.compute_psnr_ssim(
            B2A_opunet_i.numpy(), A_i.numpy())
        B2A_list = [B2A_i, B2A_op_i, B2A_unet_i, B2A_opunet_i]
        B2A2B_list = [B2A2B_i, B2A2B_op_i, B2A2B_unet_i, B2A2B_opunet_i]
        psnr_list = [psnr, psnr_op, psnr_unet, psnr_opunet]
        ssim_list = [ssim, ssim_op, ssim_unet, ssim_opunet]

        method_list = ["convolutional",
                       "operational", "unet", "operational_unet"]
        ev.plot_all_images_B2A(B_i, B2A_list, A_i, psnr_list,
                               ssim_list, save_dir, A_img_paths_test[i], method_list)
        i += 1
