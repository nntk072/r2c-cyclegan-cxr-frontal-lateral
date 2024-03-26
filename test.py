import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl
import matplotlib.pyplot as plt

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
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
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

# model
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

# resotre
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(
    args.experiment_dir, 'checkpoints')).restore()


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
save_dir = py.join(args.experiment_dir, 'samples_testing', 'A2B')
py.mkdir(save_dir)
i = 0
for A, B in zip(A_dataset_test, B_dataset_test):
    A2B, A2B2A = sample_A2B(A)
    for A_i, A2B_i, A2B2A_i, B_i in zip(A, A2B, A2B2A, B):
        # img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B2A_i.numpy()], axis=1)
        # im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
        # img = np.concatenate([A_i.numpy(), A2B_i.numpy(), B_i.numpy()], axis=1)
        # im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
        ev.plot_images_A2B(A_i, A2B_i, B_i,
                           save_dir, A_img_paths_test[i])
        i += 1

save_dir = py.join(args.experiment_dir, 'samples_testing', 'B2A')
py.mkdir(save_dir)
i = 0
for A, B in zip(A_dataset_test, B_dataset_test):
    B2A, B2A2B = sample_B2A(B)
    for B_i, B2A_i, B2A2B_i, A_i in zip(B, B2A, B2A2B, A):
        # img = np.concatenate([B_i.numpy(), B2A_i.numpy(), B2A2B_i.numpy()], axis=1)
        # im.imwrite(img, py.join(save_dir, py.name_ext(B_img_paths_test[i])))
        # img = np.concatenate([B_i.numpy(), B2A_i.numpy(), A_i.numpy()], axis=1)
        # im.imwrite(img, py.join(save_dir, py.name_ext(B_img_paths_test[i])))
        ev.plot_images_B2A(B_i, B2A_i, A_i,
                           save_dir, B_img_paths_test[i])
        i += 1
