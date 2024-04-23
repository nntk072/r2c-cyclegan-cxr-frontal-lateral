import functools

import matplotlib.pyplot as plt
import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import module
from plot import temporary_plot, save_plot_data, save_psnr_and_ssim_data, temporary_plot_psnr_ssim
import evaluation as ev
import model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# py.arg('--dataset', default='CheXpert-v1.0-small')
py.arg('--dataset', default='')
py.arg('--plot_data_dir', default='plot_data')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=286)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
# py.arg('--epochs', type=int, default=200)
py.arg('--epochs', type=int, default=1000)
# epoch to start decaying learning rate
py.arg('--epoch_decay', type=int, default=100)
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan',
       choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
# py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_mode', default='wgan-gp',
       choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
# Order of the operational layer (q parameter).
py.arg('--q', type=int, default=3)
py.arg('--method', help='convolutional, operational, unet, anotherunet',
       default='convolutional')
args = py.args()

# output_dir
output_dir = py.join('output', args.dataset)
py.mkdir(output_dir)

# plot_dir
plot_dir = py.join('output', args.dataset, args.method, args.plot_data_dir)
py.mkdir(plot_dir)
py.mkdir(py.join(plot_dir, 'training'))
py.mkdir(py.join(plot_dir, 'validation'))

py.mkdir(py.join('output', args.dataset, args.method, 'plot_figure'))
g_loss_dir = py.join('output', args.dataset, args.method,
                     'plot_figure', 'training', 'g_loss')
d_loss_dir = py.join('output', args.dataset, args.method,
                     'plot_figure', 'training', 'd_loss')
cycle_loss_dir = py.join('output', args.dataset, args.method,
                         'plot_figure', 'training', 'cycle_loss')
id_loss_dir = py.join('output', args.dataset, args.method,
                      'plot_figure', 'training', 'id_loss')
ssim_dir = py.join('output', args.dataset, args.method,
                   'plot_figure', 'training', 'ssim')
psnr_dir = py.join('output', args.dataset, args.method,
                   'plot_figure', 'training', 'psnr')
py.mkdir(g_loss_dir)
py.mkdir(d_loss_dir)
py.mkdir(cycle_loss_dir)
py.mkdir(id_loss_dir)
py.mkdir(ssim_dir)
py.mkdir(psnr_dir)
g_loss_validation_dir = py.join(
    'output', args.dataset, args.method, 'plot_figure', 'validation', 'g_loss')
d_loss_validation_dir = py.join(
    'output', args.dataset, args.method, 'plot_figure', 'validation', 'd_loss')
cycle_loss_validation_dir = py.join(
    'output', args.dataset, args.method, 'plot_figure', 'validation', 'cycle_loss')
id_loss_validation_dir = py.join(
    'output', args.dataset, args.method, 'plot_figure', 'validation', 'id_loss')
ssim_dir_valid = py.join('output', args.dataset, args.method,
                         'plot_figure', 'validation', 'ssim')
psnr_dir_valid = py.join('output', args.dataset, args.method,
                         'plot_figure', 'validation', 'psnr')
py.mkdir(g_loss_validation_dir)
py.mkdir(d_loss_validation_dir)
py.mkdir(cycle_loss_validation_dir)
py.mkdir(id_loss_validation_dir)
py.mkdir(ssim_dir_valid)
py.mkdir(psnr_dir_valid)
# save settings
# py.args_to_yaml(py.join(output_dir, args.method, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A_img_paths = py.glob(
    py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
B_img_paths = py.glob(
    py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
A_B_dataset, len_dataset = data.make_zip_dataset(
    A_img_paths, B_img_paths, args.batch_size, args.load_size, args.crop_size, training=True, repeat=False)

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

A_B_dataset_test, _ = data.make_zip_dataset(
    A_img_paths, B_img_paths, args.batch_size, args.load_size, args.crop_size, training=False, repeat=False, shuffle=False)

A_img_paths_valid = py.glob(
    py.join(args.datasets_dir, args.dataset, 'validA'), '*.jpg'
)
B_img_paths_valid = py.glob(
    py.join(args.datasets_dir, args.dataset, 'validB'), '*.jpg'
)
A_B_dataset_valid, valid_len_dataset = data.make_zip_dataset(
    A_img_paths_valid, B_img_paths_valid, args.batch_size, args.load_size, args.crop_size, training=False, repeat=False, shuffle=False)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

model = model.model(args)
model.init(len_dataset)

tf.keras.utils.plot_model(model.G_A2B, to_file="G_A2B.png", show_shapes=True)
tf.keras.utils.plot_model(model.G_B2A, to_file="G_B2A.png", show_shapes=True)
tf.keras.utils.plot_model(model.D_A, to_file="D_A.png", show_shapes=True)
tf.keras.utils.plot_model(model.D_B, to_file="D_B.png", show_shapes=True)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================


def train_step(A, B):
    A2B, B2A, G_loss_dict = model.train_G(A, B)

    # cannot autograph `A2B_pool`
    # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    A2B = A2B_pool(A2B)
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = model.train_D(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict

# ==============================================================================
# =                                    run                                     =
# ==============================================================================


# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkDir = 'output/' + args.method + '/checkpoints/'
if not os.path.exists(checkDir):
    os.makedirs(checkDir)
checkpoint = tl.Checkpoint(dict(G_A2B=model.G_A2B,
                                G_B2A=model.G_B2A,
                                D_A=model.D_A,
                                D_B=model.D_B,
                                G_optimizer=model.G_optimizer,
                                D_optimizer=model.D_optimizer,
                                ep_cnt=ep_cnt),
                           checkDir,
                           max_to_keep=1000)
# try:  # restore checkpoint including the epoch counter
#     checkpoint.restore().assert_existing_objects_matched()
# except Exception as e:
#     print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(
    py.join(output_dir, args.method, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = py.join(output_dir, args.method, 'samples_training')
py.mkdir(sample_dir)

# valid
valid_iter = iter(A_B_dataset_valid)
valid_dir = py.join(output_dir, args.method, 'samples_valid')
py.mkdir(valid_dir)

start = 0  # start epoch number
for i in range(0, 1000):
    try:
        ssim_value_list = np.load(
            py.join(plot_dir, 'training', 'ssim_A2B_value_list_' + str(i) + '.npy'))
        start = i
    except:
        break

# Restore the checkpoint from 1 to the last epoch, save the validation plot data
checkDir = checkpoint.directory
i_train = start*len_dataset
ep_step = 1000
ep_cnt_recover = tf.Variable(
            initial_value=-1, trainable=False, dtype=tf.int64)
# for ep in range(0, ep_step + 1):
with train_summary_writer.as_default():
    for ep in range(start, ep_step):
        # Load model
        
        checkpoint_path = checkDir + '/ckpt-' + str(ep)
        tl.Checkpoint(dict(G_A2B=model.G_A2B, G_B2A=model.G_B2A, D_A=model.D_A,
                           D_B=model.D_B, ep_cnt=ep_cnt_recover), checkDir).restore(checkpoint_path)

        print('Restored epoch: ', ep_cnt_recover.numpy())

        # Train restoration step (Save the loss values for each iteration, and save the plot after 5 iterations
        # iterations, A2B_g_loss, B2A_g_loss, A2B2A_cycle_loss, B2A2B_cycle_loss, A2A_id_loss, B2B_id_loss, A_d_loss, B_d_loss = [
        # ], [], [], [], [], [], [], [], []
        iterations, ssim_value_A2B, psnr_value_A2B, ssim_value_B2A, psnr_value_B2A = [], [], [], [], []

        # Restore the loss values for the training also
        for A, B in tqdm.tqdm(A_B_dataset_test, desc='Training Epoch Loop', total=len_dataset):
            # A2B, B2A, valid_G_loss = model.valid_G(A, B)
            # valid_D_loss = model.valid_D(A, B, A2B, B2A)
            # A2B_g_loss.append(valid_G_loss['A2B_g_loss'])
            # B2A_g_loss.append(valid_G_loss['B2A_g_loss'])
            # A2B2A_cycle_loss.append(valid_G_loss['A2B2A_cycle_loss'])
            # B2A2B_cycle_loss.append(valid_G_loss['B2A2B_cycle_loss'])
            # A2A_id_loss.append(valid_G_loss['A2A_id_loss'])
            # B2B_id_loss.append(valid_G_loss['B2B_id_loss'])
            # A_d_loss.append(valid_D_loss['A_d_loss'])
            # B_d_loss.append(valid_D_loss['B_d_loss'])

            # Compute the SSIM and the PSNR of the images
            A2B, B2A, _, _ = model.sample(A, B)
            psnr_A2B, ssim_A2B = ev.compute_psnr_ssim(B[0].numpy(), A2B[0].numpy())
            psnr_B2A, ssim_B2A = ev.compute_psnr_ssim(A[0].numpy(), B2A[0].numpy())

            ssim_value_A2B.append(ssim_A2B)
            psnr_value_A2B.append(psnr_A2B)
            ssim_value_B2A.append(ssim_B2A)
            psnr_value_B2A.append(psnr_B2A)
            iterations.append(i_train)

            # Make a dictionary containing the ssim and psnr values for summary the checkpoints
            summary = {'ssim_A2B': ssim_A2B, 'psnr_A2B': psnr_A2B,
                       'ssim_B2A': ssim_B2A, 'psnr_B2A': psnr_B2A}

            # Summary
            tl.summary(summary, step=i_train, name='psnr_ssim')
            i_train += 1


        # Valid step (Save the loss values for each iteration, and save the plot after 5 iterations

        # iterations_valid, A2B_g_loss_valid, B2A_g_loss_valid, A2B2A_cycle_loss_valid, B2A2B_cycle_loss_valid, A2A_id_loss_valid, B2B_id_loss_valid, A_d_loss_valid, B_d_loss_valid = [], [], [], [], [], [], [], [], []
        iterations_valid, ssim_A2B_value_valid, psnr_A2B_value_valid, ssim_B2A_value_valid, psnr_B2A_value_valid = [], [], [], [], []
        i_valid = 0
        for A, B in tqdm.tqdm(A_B_dataset_valid, desc='Valid Epoch Loop', total=valid_len_dataset):
            # A2B, B2A, valid_G_results = model.valid_G(A, B)
            # valid_D_results = model.valid_D(A, B, A2B, B2A)
            # A2B_g_loss_valid.append(valid_G_results['A2B_g_loss_valid'])
            # B2A_g_loss_valid.append(valid_G_results['B2A_g_loss_valid'])
            # A2B2A_cycle_loss_valid.append(
            #     valid_G_results['A2B2A_cycle_loss_valid'])
            # B2A2B_cycle_loss_valid.append(
            #     valid_G_results['B2A2B_cycle_loss_valid'])
            # A2A_id_loss_valid.append(valid_G_results['A2A_id_loss_valid'])
            # B2B_id_loss_valid.append(valid_G_results['B2B_id_loss_valid'])
            # A_d_loss_valid.append(valid_D_results['A_d_loss_valid'])
            # B_d_loss_valid.append(valid_D_results['B_d_loss_valid'])

            # Compute the SSIM and the PSNR of the validation results
            A2B, B2A, _, _ = model.sample(A, B)
            psnr_A2B, ssim_A2B = ev.compute_psnr_ssim(
                B[0].numpy(), A2B[0].numpy())
            psnr_B2A, ssim_B2A = ev.compute_psnr_ssim(
                A[0].numpy(), B2A[0].numpy())
            ssim_A2B_value_valid.append(ssim_A2B)
            psnr_A2B_value_valid.append(psnr_A2B)
            ssim_B2A_value_valid.append(ssim_B2A)
            psnr_B2A_value_valid.append(psnr_B2A)
            iterations_valid.append(i_valid)
            i_valid += 1
            # Summary
            valid_summary = {'ssim_A2B_valid': ssim_A2B, 'psnr_A2B_valid': psnr_A2B,
                             'ssim_B2A_valid': ssim_B2A, 'psnr_B2A_valid': psnr_B2A}
            tl.summary(valid_summary, step=i_valid, name='psnr_ssim_valid')

        """
        # Save the loss data for each iteration into a separate file
        save_plot_data(iterations, A2B_g_loss, B2A_g_loss, A2B2A_cycle_loss,
                    B2A2B_cycle_loss, A2A_id_loss, B2B_id_loss, A_d_loss, B_d_loss, ep, "training", args.method)

        # Save the loss validation data for each iteration into a separate file
        save_plot_data(iterations_valid, A2B_g_loss_valid, B2A_g_loss_valid, A2B2A_cycle_loss_valid,
                    B2A2B_cycle_loss_valid, A2A_id_loss_valid, B2B_id_loss_valid, A_d_loss_valid, B_d_loss_valid, ep, "validation", args.method)

        temporary_plot(g_loss_dir, d_loss_dir, cycle_loss_dir, id_loss_dir, iterations, A2B_g_loss,
                    B2A_g_loss, A2B2A_cycle_loss, B2A2B_cycle_loss, A2A_id_loss, B2B_id_loss, A_d_loss, B_d_loss, ep)

        temporary_plot(g_loss_validation_dir, d_loss_validation_dir, cycle_loss_validation_dir, id_loss_validation_dir, iterations_valid, A2B_g_loss_valid,
                    B2A_g_loss_valid, A2B2A_cycle_loss_valid, B2A2B_cycle_loss_valid, A2A_id_loss_valid, B2B_id_loss_valid, A_d_loss_valid, B_d_loss_valid, ep)
        """

        # Training dataset handling
        save_psnr_and_ssim_data(iterations, ssim_value_A2B, psnr_value_A2B,
                                ssim_value_B2A, psnr_value_B2A, ep, "training", args.method)
        temporary_plot_psnr_ssim(
            ssim_dir, psnr_dir, iterations, ssim_value_A2B, psnr_value_A2B, ssim_value_B2A, psnr_value_B2A, ep)

        # Validation dataset handling
        save_psnr_and_ssim_data(iterations_valid, ssim_A2B_value_valid, psnr_A2B_value_valid,
                                ssim_B2A_value_valid, psnr_B2A_value_valid, ep, "validation", args.method)
        temporary_plot_psnr_ssim(ssim_dir_valid, psnr_dir_valid,
                                 iterations_valid, ssim_A2B_value_valid, psnr_A2B_value_valid, ssim_B2A_value_valid, psnr_B2A_value_valid, ep)
