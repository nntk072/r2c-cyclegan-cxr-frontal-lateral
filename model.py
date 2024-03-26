import tensorflow as tf

import tf2gan as gan
import functools
import module


class model:
    def __init__(self,args):
        self.G_A2B = None
        self.G_B2A = None

        self.D_A = None
        self.D_B = None
        self.args = args
        self.d_loss_fn, self.g_loss_fn = gan.get_adversarial_losses_fn(
    args.adversarial_loss_mode)
        self.cycle_loss_fn = tf.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.losses.MeanAbsoluteError()

        self.G_lr_scheduler = None
        self.D_lr_scheduler = None
        self.G_optimizer = None
        self.D_optimizer = None

        self.cycle_weights = None
        self.identity_weight = None

        self.filter = None

    def init(self, len_dataset):

        self.filter = self.args.method

        self.cycle_weights = self.args.cycle_loss_weight
        self.identity_weight = self.args.identity_loss_weight
        self.G_lr_scheduler = module.LinearDecay(
            self.args.lr, self.args.epochs * len_dataset, self.args.epoch_decay * len_dataset)
        self.D_lr_scheduler = module.LinearDecay(
            self.args.lr, self.args.epochs * len_dataset, self.args.epoch_decay * len_dataset)
        self.G_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.G_lr_scheduler, beta_1=self.args.beta_1)
        self.D_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.D_lr_scheduler, beta_1=self.args.beta_1)

        # Creating models.
        if self.args.method == 'operational':
            self.set_G_A2B(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=self.args.q)
            self.set_G_B2A(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=self.args.q)
            self.set_D_A(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=self.args.q)
            self.set_D_B(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=self.args.q)
        elif self.args.method == 'convolutional':
            self.set_G_A2B(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=None)
            self.set_G_B2A(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=None)
            self.set_D_A(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=None)
            self.set_D_B(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=None)
        elif self.args.method == 'unet':
            self.set_G_A2B(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=None)
            self.set_G_B2A(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=None)
            self.set_D_A(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=None)
            self.set_D_B(input_shape=(
                self.args.crop_size, self.args.crop_size, 3), q=None)
        else:
            print('Undefined filtering method!')

    def set_G_A2B(self, input_shape, q):
        if self.args.method == 'operational':
            self.G_A2B = module.OpGenerator(input_shape=input_shape, q=q)
        elif self.args.method == 'convolutional':
            self.G_A2B = module.ResnetGenerator(input_shape=input_shape)
        elif self.args.method == 'unet':
            self.G_A2B = module.UNetGenerator(input_shape=input_shape)
        else:
            print('Undefined filtering method!')

    def set_G_B2A(self, input_shape, q):
        if self.args.method == 'operational':
            self.G_B2A = module.OpGenerator(input_shape=input_shape, q=q)
        elif self.args.method == 'convolutional':
            self.G_B2A = module.ResnetGenerator(input_shape=input_shape)
        elif self.args.method == 'unet':
            self.G_B2A = module.UNetGenerator(input_shape=input_shape)
        else:
            print('Undefined filtering method!')

    def set_D_A(self, input_shape, q):
        if self.args.method == 'operational':
            self.D_A = module.OpDiscriminator(input_shape=input_shape, q=q)
        elif self.args.method == 'convolutional':
            self.D_A = module.ConvDiscriminator(input_shape=input_shape)
        elif self.args.method == 'unet':
            self.D_A = module.UNetDiscriminator(input_shape=input_shape)
        else:
            print('Undefined filtering method!')

    def set_D_B(self, input_shape, q):
        if self.filter == 'operational':
            self.D_B = module.OpDiscriminator(input_shape=input_shape, q=q)
        elif self.filter == 'convolutional':
            self.D_B = module.ConvDiscriminator(input_shape=input_shape)
        elif self.filter == 'unet':
            self.D_B = module.UNetDiscriminator(input_shape=input_shape)
        else:
            print('Undefined filtering method!')

    @tf.function
    def train_G(self, A, B):
        with tf.GradientTape() as t:
            A2B = self.G_A2B(A, training=True)  # label_A
            B2A = self.G_B2A(B, training=True)  # label_B
            A2B2A = self.G_B2A(A2B, training=True)  # label_A
            B2A2B = self.G_A2B(B2A, training=True)  # label_B
            A2A = self.G_B2A(A, training=True)  # label_A
            B2B = self.G_A2B(B, training=True)  # label_B

            A2B_d_logits = self.D_B(A2B, training=True)
            B2A_d_logits = self.D_A(B2A, training=True)

            A2B_g_loss = self.g_loss_fn(A2B_d_logits)
            B2A_g_loss = self.g_loss_fn(B2A_d_logits)
            A2B2A_cycle_loss = self.cycle_loss_fn(A, A2B2A)
            B2A2B_cycle_loss = self.cycle_loss_fn(B, B2A2B)
            A2A_id_loss = self.identity_loss_fn(A, A2A)
            B2B_id_loss = self.identity_loss_fn(B, B2B)

            G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * self.cycle_weights + (
                A2A_id_loss + B2B_id_loss) * self.identity_weight

        G_grad = t.gradient(
            G_loss, self.G_A2B.trainable_variables + self.G_B2A.trainable_variables)
        self.G_optimizer.apply_gradients(
            zip(G_grad, self.G_A2B.trainable_variables + self.G_B2A.trainable_variables))

        return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                          'B2A_g_loss': B2A_g_loss,
                          'A2B2A_cycle_loss': A2B2A_cycle_loss,
                          'B2A2B_cycle_loss': B2A2B_cycle_loss,
                          'A2A_id_loss': A2A_id_loss,
                          'B2B_id_loss': B2B_id_loss}

    @tf.function
    def train_D(self, A, B, A2B, B2A):
        with tf.GradientTape() as t:
            A_d_logits = self.D_A(A, training=True)
            B2A_d_logits = self.D_A(B2A, training=True)
            B_d_logits = self.D_B(B, training=True)
            A2B_d_logits = self.D_B(A2B, training=True)

            A_d_loss, B2A_d_loss = self.d_loss_fn(A_d_logits, B2A_d_logits)
            B_d_loss, A2B_d_loss = self.d_loss_fn(B_d_logits, A2B_d_logits)

            D_A_gp = gan.gradient_penalty(functools.partial(
                self.D_A, training=True), A, B2A, mode=self.args.gradient_penalty_mode)
            D_B_gp = gan.gradient_penalty(functools.partial(
                self.D_B, training=True), B, A2B, mode=self.args.gradient_penalty_mode)

            D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss)

        D_grad = t.gradient(
            D_loss, self.D_A.trainable_variables + self.D_B.trainable_variables)
        self.D_optimizer.apply_gradients(
            zip(D_grad, self.D_A.trainable_variables + self.D_B.trainable_variables))
        return {'A_d_loss': A_d_loss + B2A_d_loss,
                'B_d_loss': B_d_loss + A2B_d_loss,
                'D_A_gp': D_A_gp,
                'D_B_gp': D_B_gp}

    @tf.function
    def valid_G(self, A, B):
        A2B = self.G_A2B(A, training=False)
        B2A = self.G_B2A(B, training=False)
        A2B2A = self.G_B2A(A2B, training=False)
        B2A2B = self.G_A2B(B2A, training=False)
        A2A = self.G_B2A(A, training=False)
        B2B = self.G_A2B(B, training=False)

        A2B_d_logits = self.D_B(A2B, training=False)
        B2A_d_logits = self.D_A(B2A, training=False)

        A2B_g_loss = self.g_loss_fn(A2B_d_logits)
        B2A_g_loss = self.g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = self.cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = self.cycle_loss_fn(B, B2A2B)
        A2A_id_loss = self.identity_loss_fn(A, A2A)
        B2B_id_loss = self.identity_loss_fn(B, B2B)

        return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                        'B2A_g_loss': B2A_g_loss,
                        'A2B2A_cycle_loss': A2B2A_cycle_loss,
                        'B2A2B_cycle_loss': B2A2B_cycle_loss,
                        'A2A_id_loss': A2A_id_loss,
                        'B2B_id_loss': B2B_id_loss}
    @tf.function
    def valid_D(self, A, B, A2B, B2A):
        A_d_logits = self.D_A(A, training=False)
        B2A_d_logits = self.D_A(B2A, training=False)
        B_d_logits = self.D_B(B, training=False)
        A2B_d_logits = self.D_B(A2B, training=False)

        A_d_loss, B2A_d_loss = self.d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = self.d_loss_fn(B_d_logits, A2B_d_logits)
        D_A_gp = gan.gradient_penalty(functools.partial(
            self.D_A, training=False), A, B2A, mode=self.args.gradient_penalty_mode)
        D_B_gp = gan.gradient_penalty(functools.partial(
            self.D_B, training=False), B, A2B, mode=self.args.gradient_penalty_mode)

        return {'A_d_loss': A_d_loss + B2A_d_loss,
                'B_d_loss': B_d_loss + A2B_d_loss,
                'D_A_gp': D_A_gp,
                'D_B_gp': D_B_gp}
    
    @tf.function
    def sample(self, A, B):
        A2B = self.G_A2B(A, training=False)
        B2A = self.G_B2A(B, training=False)
        A2B2A = self.G_B2A(A2B, training=False)
        B2A2B = self.G_A2B(B2A, training=False)
        return A2B, B2A, A2B2A, B2A2B
