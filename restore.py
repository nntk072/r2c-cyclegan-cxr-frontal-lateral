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
from plot import temporary_plot, save_plot_data
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
args = py.args()

# output_dir
output_dir = py.join('output', args.dataset)
py.mkdir(output_dir)

# plot_dir
plot_dir = py.join('output', args.dataset, args.plot_data_dir)
py.mkdir(plot_dir)
py.mkdir(py.join(plot_dir, 'training'))
py.mkdir(py.join(plot_dir, 'validation'))

# py.mkdir(py.join('output', args.dataset, 'plot_figure'))
# g_loss_dir = py.join('output', args.dataset, 'plot_figure', 'g_loss')
# d_loss_dir = py.join('output', args.dataset, 'plot_figure', 'd_loss')
# cycle_loss_dir = py.join('output', args.dataset, 'plot_figure', 'cycle_loss')
# id_loss_dir = py.join('output', args.dataset, 'plot_figure', 'id_loss')
# py.mkdir(g_loss_dir)
# py.mkdir(d_loss_dir)
# py.mkdir(cycle_loss_dir)
# py.mkdir(id_loss_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


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
    A_img_paths, B_img_paths, args.batch_size, args.load_size, args.crop_size, training=False, repeat=False)

A_img_paths_valid = py.glob(
    py.join(args.datasets_dir, args.dataset, 'validA'), '*.jpg'
)
B_img_paths_valid = py.glob(
    py.join(args.datasets_dir, args.dataset, 'validB'), '*.jpg'
)
A_B_dataset_valid, _ = data.make_zip_dataset(
    A_img_paths_valid, B_img_paths_valid, args.batch_size, args.load_size, args.crop_size, training=False, repeat=False)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(
    args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = module.LinearDecay(
    args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(
    args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(
    learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(
    learning_rate=D_lr_scheduler, beta_1=args.beta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B):
    with tf.GradientTape() as t:
        A2B = G_A2B(A, training=True)
        B2A = G_B2A(B, training=True)
        A2B2A = G_B2A(A2B, training=True)
        B2A2B = G_A2B(B2A, training=True)
        A2A = G_B2A(A, training=True)
        B2B = G_A2B(B, training=True)

        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * \
            args.cycle_loss_weight + \
            (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables +
                        G_B2A.trainable_variables)
    G_optimizer.apply_gradients(
        zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss}


@tf.function
def train_D(A, B, A2B, B2A):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        B2A_d_logits = D_A(B2A, training=True)
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        D_A_gp = gan.gradient_penalty(functools.partial(
            D_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
        D_B_gp = gan.gradient_penalty(functools.partial(
            D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + \
            (D_A_gp + D_B_gp) * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_A.trainable_variables +
                        D_B.trainable_variables)
    D_optimizer.apply_gradients(
        zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}


@tf.function
def valid_G(A, B):
    A2B = G_A2B(A, training=False)
    B2A = G_B2A(B, training=False)
    A2B2A = G_B2A(A2B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    A2A = G_B2A(A, training=False)
    B2B = G_A2B(B, training=False)

    A2B_d_logits = D_B(A2B, training=False)
    B2A_d_logits = D_A(B2A, training=False)

    A2B_g_loss = g_loss_fn(A2B_d_logits)
    B2A_g_loss = g_loss_fn(B2A_d_logits)
    A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
    B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
    A2A_id_loss = identity_loss_fn(A, A2A)
    B2B_id_loss = identity_loss_fn(B, B2B)

    return {'A2B_g_loss': A2B_g_loss,
            'B2A_g_loss': B2A_g_loss,
            'A2B2A_cycle_loss': A2B2A_cycle_loss,
            'B2A2B_cycle_loss': B2A2B_cycle_loss,
            'A2A_id_loss': A2A_id_loss,
            'B2B_id_loss': B2B_id_loss}


@tf.function
def valid_D(A, B, A2B, B2A):
    A_d_logits = D_A(A, training=False)
    B2A_d_logits = D_A(B2A, training=False)
    B_d_logits = D_B(B, training=False)
    A2B_d_logits = D_B(A2B, training=False)

    A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
    B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
    D_A_gp = gan.gradient_penalty(functools.partial(
        D_A, training=False), A, B2A, mode=args.gradient_penalty_mode)
    D_B_gp = gan.gradient_penalty(functools.partial(
        D_B, training=False), B, A2B, mode=args.gradient_penalty_mode)

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}


def train_step(A, B):
    A2B, B2A, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    A2B = A2B_pool(A2B)
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    A2B = G_A2B(A, training=False)
    B2A = G_B2A(B, training=False)
    A2B2A = G_B2A(A2B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return A2B, B2A, A2B2A, B2A2B


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_B2A=G_B2A,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=1000)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(
    py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# valid
valid_iter = iter(A_B_dataset_test)
valid_dir = py.join(output_dir, 'samples_valid')
py.mkdir(valid_dir)


# Restore the checkpoint from 1 to the last epoch, save the validation plot data
checkDir = checkpoint.directory
ep_step = 1000
for ep in range(0, ep_step + 1):
    # Load model
    # try:
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    checkpoint_path = "ckpt-"+str(ep)
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A, D_A=D_A,
                        D_B=D_B, ep_cnt=ep_cnt), checkDir).restore(checkpoint_path)
    # except:
    #     break

    print('Restored epoch: ', ep_cnt.numpy())
    i = 0

    # Train restoration step (Save the loss values for each iteration, and save the plot after 5 iterations
    iterations, A2B_g_loss, B2B_g_loss, A2B2A_cycle_loss, B2A2B_cycle_loss, A2A_id_loss, B2B_id_loss, A_d_loss, B_d_loss = [
    ], [], [], [], [], [], [], [], []

    # Valid step (Save the loss values for each iteration, and save the plot after 5 iterations
    iterations_valid, A2B_g_loss_valid, B2A_g_loss_valid, A2B2A_cycle_loss_valid, B2A2B_cycle_loss_valid, A2A_id_loss_valid, B2B_id_loss_valid, A_d_loss_valid, B_d_loss_valid = [], [], [], [], [], [], [], [], []
    for A, B in tqdm.tqdm(A_B_dataset_valid, desc='Valid Epoch Loop', total=len_dataset):
        A2B_g_loss_valid.append(valid_G(A, B)['A2B_g_loss'])
        B2A_g_loss_valid.append(valid_G(A, B)['B2A_g_loss'])
        A2B2A_cycle_loss_valid.append(valid_G(A, B)['A2B2A_cycle_loss'])
        B2A2B_cycle_loss_valid.append(valid_G(A, B)['B2A2B_cycle_loss'])
        A2A_id_loss_valid.append(valid_G(A, B)['A2A_id_loss'])
        B2B_id_loss_valid.append(valid_G(A, B)['B2B_id_loss'])
        A_d_loss_valid.append(valid_D(A, B, A2B, B2A)['A_d_loss'])
        B_d_loss_valid.append(valid_D(A, B, A2B, B2A)['B_d_loss'])
        iterations_valid.append(G_optimizer.iterations.numpy())

    if ep != 0 and (ep-1) % 5 == 0:
        A, B = next(valid_iter)
        A2B, B2A, A2B2A, B2A2B = sample(A, B)
        img = im.immerge(np.concatenate(
            [A, A2B, B, B2A], axis=0), n_rows=2)
        im.imwrite(img, py.join(valid_dir, 'iter-%09d.jpg' %
                                G_optimizer.iterations.numpy()))

        A, B = next(test_iter)
        A2B, B2A, A2B2A, B2A2B = sample(A, B)
        img = im.immerge(np.concatenate(
            [A, A2B, B, B2A], axis=0), n_rows=2)
        im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' %
                                G_optimizer.iterations.numpy()))

    # Save the loss validation data for each iteration into a separate file
    save_plot_data(iterations_valid, A2B_g_loss_valid, B2A_g_loss_valid, A2B2A_cycle_loss_valid,
                   B2A2B_cycle_loss_valid, A2A_id_loss_valid, B2B_id_loss_valid, A_d_loss_valid, B_d_loss_valid, ep, "validation")
