import numpy as np
import tensorflow as tf
import utils
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *


class ResnetBlock_change_dim(Layer):
    def __init__(self, input_dim, output_dim, stride=[1, 1], padding_type='same', norm_layer=BatchNormalization,
                 use_dropout=False, use_bias=False, named = None):
        super(ResnetBlock_change_dim, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stride_list = stride
        self.named = named
        self.conv_block = self.build_conv_block(padding_type, norm_layer, use_dropout, use_bias)
        if self.input_dim != self.output_dim or self.stride_list[0] != self.stride_list[1]:
            self.change_dim_conv = Sequential()
            if self.named is not None:
                self.change_dim_conv.add(Conv2D(filters=self.output_dim, kernel_size=1,
                                         strides=self.stride_list[0] * self.stride_list[1],
                                         name=self.named + '_change_channel'))
            else:
                self.change_dim_conv.add(Conv2D(filters=self.output_dim, kernel_size=1,
                                         strides=self.stride_list[0] * self.stride_list[1]))
            self.change_dim_conv.add(norm_layer(axis=-1))

    def build_conv_block(self, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = Sequential()
        if self.named is not None:
            conv_block.add(
                Conv2D(filters=self.output_dim, strides=self.stride_list[0], kernel_size=3, padding=padding_type,
                       use_bias=use_bias, name=self.named+'_conv1'))
            conv_block.add(norm_layer(axis=-1, name=self.named + '_BN1'))
        else:
            conv_block.add(Conv2D(filters=self.output_dim, strides=self.stride_list[0], kernel_size=3, padding=padding_type, use_bias=use_bias))
            conv_block.add(norm_layer(axis=-1))
        conv_block.add(LeakyReLU())

        if use_dropout:
            conv_block.add(Dropout(0.5))

        if self.named is not None:
            conv_block.add(
                Conv2D(filters=self.output_dim, strides=self.stride_list[1], kernel_size=3, padding=padding_type,
                       use_bias=use_bias, name=self.named+'_conv2'))
            conv_block.add(norm_layer(axis=-1, name=self.named + '_BN2'))
        else:
            conv_block.add(Conv2D(filters=self.output_dim, strides=self.stride_list[1], kernel_size=3, padding=padding_type,use_bias=use_bias))
            conv_block.add(norm_layer(axis=-1))

        return conv_block

    def call(self, x):
        if self.input_dim != self.output_dim or self.stride_list[0] != self.stride_list[1]:
            out = self.change_dim_conv(x) + self.conv_block(x)
        else:
            out = x + self.conv_block(x)
        return out


class ChartStampEncoder(Layer):

    def __init__(self):
        super(ChartStampEncoder, self).__init__()
        self.secret_dense = Dense(19200, kernel_initializer='he_normal', name='secret_dense')  #for 1000/10000 bit
        # self.secret_dense = Dense(7500, kernel_initializer='he_normal', name='secret_dense') #for 100 bit

        self.block1 = Sequential([
            Conv2D(32, 7, padding='same', name='block1_conv1'),
            BatchNormalization(name='block1_bn1'),
            LeakyReLU()
        ])

        self.block2 = ResnetBlock_change_dim(32, 32, stride=[2, 1], named='block2')

        self.block3 = ResnetBlock_change_dim(32, 64, stride=[2, 1], named='block3')

        self.block4 = ResnetBlock_change_dim(64, 128, stride=[2, 1], named='block4')

        self.block5 = ResnetBlock_change_dim(128, 256, stride=[2, 1], named='block5')

        self.up5 = Sequential([
            Conv2D(128, 3, padding='same', name='up5'),
            BatchNormalization(name='up5_bn'),
            LeakyReLU()
        ])

        self.block6 = ResnetBlock_change_dim(256, 128, named='block6')

        self.up6 = Sequential([
            Conv2D(64, 3, padding='same', name='up6'),
            BatchNormalization(name='up6_bn'),
            LeakyReLU()
        ])

        self.block7 = ResnetBlock_change_dim(128, 64, named='block7')

        self.up7 = Sequential([
            Conv2D(32, 3, padding='same', name='up7'),
            BatchNormalization(name='up7_bn'),
            LeakyReLU()
        ])

        self.block8 = ResnetBlock_change_dim(64, 32, named='block8')

        self.up8 = Sequential([
            Conv2D(32, 3, padding='same', name='up8'),
            BatchNormalization(name='up7_bn'),
            LeakyReLU()
        ])
        self.block9 = ResnetBlock_change_dim(70, 32, named='block9')
        self.residual = Conv2D(3, 1, activation=None, padding='same', name='up9')

    def call(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)

        secret = Reshape((80, 80, 3))(secret)    #for 1000/10000 bit
        secret_enlarged = UpSampling2D(size=(5, 5))(secret)

        # secret = Reshape((50, 50, 3))(secret)   #for 100 bit
        # secret_enlarged = UpSampling2D(size=(8, 8))(secret)

        inputs = concatenate([secret_enlarged, image], axis=-1)
        block1 = self.block1(inputs)

        block2 = self.block2(block1)

        block3 = self.block3(block2)

        block4 = self.block4(block3)

        block5 = self.block5(block4)

        up5 = self.up5(UpSampling2D(size=(2, 2))(block5))
        merge5 = concatenate([block4, up5], axis=-1)
        block6 = self.block6(merge5)
        up6 = self.up6(UpSampling2D(size=(2, 2))(block6))
        merge6 = concatenate([block3, up6], axis=-1)
        block7 = self.block7(merge6)
        up7 = self.up7(UpSampling2D(size=(2, 2))(block7))
        merge7 = concatenate([block2, up7], axis=-1)
        block8 = self.block8(merge7)
        up8 = self.up8(UpSampling2D(size=(2, 2))(block8))
        merge8 = concatenate([inputs, block1, up8], axis=-1)
        block9 = self.block9(merge8)
        residual = self.residual(block9)

        return residual


class SimpleChartStampEncoder(Layer):  #simple encoder for 100 bit

    def __init__(self):
        super(SimpleChartStampEncoder, self).__init__()
        self.secret_dense = Dense(7500, activation='relu', kernel_initializer='he_normal')
        self.conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(32, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv4 = Conv2D(128, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv5 = Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.residual = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        secret = Reshape((50, 50, 3))(secret)
        secret_enlarged = UpSampling2D(size=(8, 8))(secret)

        inputs = concatenate([secret_enlarged, image], axis=-1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(UpSampling2D(size=(2,2))(conv5))
        merge6 = concatenate([conv4,up6], axis=3)
        conv6 = self.conv6(merge6)
        up7 = self.up7(UpSampling2D(size=(2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis=3)
        conv7 = self.conv7(merge7)
        up8 = self.up8(UpSampling2D(size=(2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis=3)
        conv8 = self.conv8(merge8)
        up9 = self.up9(UpSampling2D(size=(2,2))(conv8))
        merge9 = concatenate([conv1,up9,inputs], axis=3)
        conv9 = self.conv9(merge9)
        # conva = self.conv9(merge9)
        # conv10 = self.conv10(conv9)
        residual = self.residual(conv9)
        return residual


class ChartStampDecoder(Layer):
    def __init__(self, secret_size):
        super(ChartStampDecoder, self).__init__()

        self.decoder = Sequential([
            ResnetBlock_change_dim(3, 32, stride=[2, 1], named='decoder_block1'),
            ResnetBlock_change_dim(32, 64, stride=[2, 1], named='decoder_block2'),
            ResnetBlock_change_dim(64, 64, stride=[2, 1], named='decoder_block2'),
            ResnetBlock_change_dim(64, 64, stride=[2, 1], named='decoder_block3'),
            Flatten(),
            # Dense(512, activation='relu', name='decoder_fc1'),  #for 100 bit encode
            Dense(4096, activation='relu', name='decoder_fc1'), #for 1000 bit encode
                                                                #for 10000 bit encode, not need additional layer
            Dense(secret_size, name='decoder_fc2')
        ])

    def call(self, image):
        image = image - .5
        return self.decoder(image)


class SimpleChartStampDecoder(Layer):  #simple encoder for 100 bit
    def __init__(self, secret_size):
        super(SimpleChartStampDecoder, self).__init__()

        self.decoder = Sequential([
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(secret_size)
        ])

    def call(self, image):
        image = image - .5
        return self.decoder(image)


class Discriminator(Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = Sequential([
            Conv2D(32, 7, padding='same', name='dis_conv1', strides=2),
            BatchNormalization(name='dis_bn1'),
            LeakyReLU(),
            ResnetBlock_change_dim(32, 32, stride=[2, 1], named='dis_block1'),
            ResnetBlock_change_dim(32, 64, stride=[2, 1], named='dis_block2'),
            ResnetBlock_change_dim(64, 128, stride=[2, 1], named='dis_block3'),
            ResnetBlock_change_dim(128, 256, stride=[2, 1], named='dis_block4'),
            Flatten(),
            Dense(1, name='dis_fc')
        ])

    def call(self, image):
            x = image - .5
            x = self.model(x)
            output = tf.reduce_mean(x)
            return output, x


class SimpleDiscriminator(Layer):  #for 100 bit
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        self.model = Sequential([
            Conv2D(8, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(16, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(1, (3, 3), activation=None, padding='same')
        ])

    def call(self, image):
            x = image - .5
            x = self.model(x)
            output = tf.reduce_mean(x)
            return output, x


def transform_net(encoded_image, args, M, global_step):

    ramp_fn = lambda ramp : tf.minimum(tf.to_float(global_step) / ramp, 1.)

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = utils.get_rnd_brightness_tf(rnd_bri, rnd_hue, tf.shape(encoded_image)[0])

    jpeg_quality = 100. - tf.random.uniform([]) * ramp_fn(args.jpeg_quality_ramp) * (100.-args.jpeg_quality)
    jpeg_factor = tf.cond(tf.less(jpeg_quality, 50), lambda: 5000. / jpeg_quality, lambda: 200. - jpeg_quality * 2) / 100. + .0001

    rnd_noise = tf.random.uniform([]) * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = tf.random.uniform([]) * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    #perspective transform
    encoded_image_warped = tf.contrib.image.transform(encoded_image, M[:,1,:], interpolation='BILINEAR')
    mask = tf.contrib.image.transform(tf.ones_like(encoded_image), M[:, 1, :], interpolation='BILINEAR')
    encoded_image_warped += (1 - mask) * tf.ones_like(encoded_image)

    # blur
    f = utils.random_blur_kernel(probs=[.25,.25], N_blur=7,
                           sigrange_gauss=[1.,3.], sigrange_line=[.25,1.], wmin_line=3)
    encoded_image = tf.nn.conv2d(encoded_image, f, [1,1,1,1], padding='SAME')

    #noise
    noise = tf.random_normal(shape=tf.shape(encoded_image), mean=0.0, stddev=rnd_noise, dtype=tf.float32)
    encoded_image_tmp = encoded_image
    encoded_image = tf.where(encoded_image + noise <= 1., encoded_image + noise, encoded_image)
    encoded_image = tf.where(encoded_image >= 0., encoded_image, encoded_image_tmp)

    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    #contrast + brightness then scale to valid range
    contrast_scale = tf.random_uniform(shape=[tf.shape(encoded_image)[0]], minval=contrast_params[0], maxval=contrast_params[1])
    contrast_scale = tf.reshape(contrast_scale, shape=[tf.shape(encoded_image)[0],1,1,1])
    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness

    low_bound = tf.reduce_min(encoded_image, axis=1,keep_dims=True)
    low_bound = tf.reduce_min(low_bound, axis=2, keep_dims=True)

    low_bound_clip = tf.maximum(low_bound, 0.)

    upper_bound = tf.reduce_max(encoded_image, axis=1, keep_dims=True)
    upper_bound = tf.reduce_max(upper_bound, axis=2, keep_dims=True)
    upper_bound_clip = tf.minimum(upper_bound, 1.)

    rate = (upper_bound_clip-low_bound_clip)/(upper_bound - low_bound)

    encoded_image = (encoded_image-low_bound) * rate + low_bound_clip
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    #lum
    encoded_image_lum = tf.expand_dims(tf.reduce_sum(encoded_image * tf.constant([.3,.6,.1]), axis=3), 3)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    encoded_image = tf.reshape(encoded_image, [-1,400,400,3])
    if not args.no_jpeg:
        encoded_image = utils.jpeg_compress_decompress(encoded_image, rounding=utils.round_only_at_0, factor=jpeg_factor, downsample_c=True)

    return encoded_image


def get_secret_acc(secret_true,secret_pred):
    with tf.variable_scope("acc"):
        secret_pred = tf.round(tf.sigmoid(secret_pred))
        correct_pred = tf.to_int64(tf.shape(secret_pred)[1]) - tf.count_nonzero(secret_pred - secret_true, axis=1)
        bit_acc = tf.reduce_sum(correct_pred) / tf.size(secret_pred, out_type=tf.int64)
        return bit_acc


def build_model(encoder,
                decoder,
                discriminator,
                secret_input,
                image_input,
                M,
                loss_scales,
                yuv_scales,
                seg_pro,
                args,
                global_step):

    residual = encoder((secret_input, image_input))
    encoded_image = image_input + residual
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    D_output_real, _ = discriminator(image_input)
    D_output_fake, D_heatmap = discriminator(encoded_image)
    transformed_image = transform_net(encoded_image, args, M, global_step)

    decoded_secret = decoder(transformed_image)

    bit_acc = get_secret_acc(secret_input, decoded_secret)

    secret_loss_op = tf.losses.sigmoid_cross_entropy(secret_input, decoded_secret)

    # if use sobel kernel to calculate image smoothness,use kernel_x and kernel_y
    kernel_x = tf.constant([
        [  # R_1  R_2 R_3   # G_1 G_2 G_3   # B_1 B_2 B_3
            [[1. / 4., 0., 0.], [0., 1. / 4., 0.], [0., 0., 1. / 4.]],  # width_1
            [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],  # width_2
            [[-1. / 4., 0., 0.], [0., -1. / 4., 0.], [0., 0., -1. / 4.]]  # width_3
        ],  # height_1
        [
            [[1. / 2., 0., 0.], [0., 1. / 2., 0.], [0., 0., 1. / 2.]],  # width_1
            [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],  # width_2
            [[-1. / 2., 0., 0.], [0., -1. / 2., 0.], [0., 0., -1. / 2.]]  # width_3
        ],  # height_2
        [
            [[1. / 4., 0., 0.], [0., 1. / 4., 0.], [0., 0., 1. / 4.]],  # width_1
            [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],  # width_2
            [[-1. / 4., 0., 0.], [0., -1. / 4., 0.], [0., 0., -1. / 4.]]  # width_3
        ]  # height_3
    ])

    kernel_y = tf.constant([
        [  # R_1  R_2 R_3   # G_1 G_2 G_3   # B_1 B_2 B_3
            [[-1. / 4., 0., 0.], [0., -1. / 4., 0.], [0., 0., -1. / 4.]],  # width_1
            [[-1. / 2., 0., 0.], [0., -1. / 2., 0.], [0., 0., -1. / 2.]],  # width_2
            [[-1. / 4., 0., 0.], [0., -1. / 4., 0.], [0., 0., -1. / 4.]]  # width_3
        ],  # height_1
        [
            [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],  # width_1
            [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],  # width_2
            [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]  # width_3
        ],  # height_2
        [
            [[1. / 4., 0., 0.], [0., 1. / 4., 0.], [0., 0., 1. / 4.]],  # width_1
            [[1. / 2., 0., 0.], [0., 1. / 2., 0.], [0., 0., 1. / 2.]],  # width_2
            [[1. / 4., 0., 0.], [0., 1. / 4., 0.], [0., 0., 1. / 4.]]  # width_3
        ]  # height_3
    ])
    v = 1.0 / 9.0
    kernel_blur = tf.constant([
        [  # R_1  R_2 R_3   # G_1 G_2 G_3   # B_1 B_2 B_3
            [[v, 0., 0.], [0., v, 0.], [0., 0., v]],  # width_1
            [[v, 0., 0.], [0., v, 0.], [0., 0., v]],  # width_2
            [[v, 0., 0.], [0., v, 0.], [0., 0., v]]  # width_3
        ],  # height_1
        [
            [[v, 0., 0.], [0., v, 0.], [0., 0., v]],  # width_1
            [[v, 0., 0.], [0., v, 0.], [0., 0., v]],  # width_2
            [[v, 0., 0.], [0., v, 0.], [0., 0., v]]  # width_3
        ],  # height_2
        [
            [[v, 0., 0.], [0., v, 0.], [0., 0., v]],  # width_1
            [[v, 0., 0.], [0., v, 0.], [0., 0., v]],  # width_2
            [[v, 0., 0.], [0., v, 0.], [0., 0., v]]  # width_3
        ]  # height_3
    ])
    encoded_image_yuv = tf.image.rgb_to_yuv(encoded_image)
    image_input_yuv = tf.image.rgb_to_yuv(image_input)
    im_diff = encoded_image_yuv-image_input_yuv

    alpha = tf.constant(1.0)
    yuv_loss_op = tf.reduce_mean(tf.square(im_diff) * (1.0+seg_pro*alpha), axis=[0, 1, 2])

    #L_R loss
    image_loss_op = tf.tensordot(yuv_loss_op, yuv_scales, axes=1)

    #L_S loss
    image_blur = image_input_yuv
    image_gradient_x = tf.nn.conv2d(image_blur, kernel_x, [1, 1, 1, 1], padding="SAME")
    image_gradient_y = tf.nn.conv2d(image_blur, kernel_y, [1, 1, 1, 1], padding="SAME")
    image_gradient = tf.sqrt(tf.square(image_gradient_x) + tf.square(image_gradient_y))

    b = tf.constant(0.05)
    a = tf.constant(1.0)
    image_weight = tf.div(a, tf.add(image_gradient, b))
    encoded_blur = tf.nn.conv2d(im_diff, kernel_blur, [1, 1, 1, 1], padding="SAME")
    encoded_gradient_x = tf.nn.conv2d(encoded_blur, kernel_x, [1, 1, 1, 1], padding="SAME")
    encoded_gradient_y = tf.nn.conv2d(encoded_blur, kernel_y, [1, 1, 1, 1], padding="SAME")
    encoded_gradient = tf.sqrt(tf.square(encoded_gradient_x) + tf.square(encoded_gradient_y))
    zero = tf.zeros_like(encoded_gradient)
    encoded_gradient = tf.where(encoded_gradient < 0.0008, zero, encoded_gradient)
    one = tf.ones([tf.shape(seg_pro)[0], 400, 400, 3])
    thre = seg_pro*one

    blur_loss_pre = tf.where(thre > 0.9, encoded_gradient * tf.div(a, b) * (1.0+seg_pro*alpha),
                             encoded_gradient * image_weight * (1.0+seg_pro*alpha))
    yuv_blur_loss_op = tf.reduce_mean(blur_loss_pre, axis=[0, 1, 2])
    image_blur_loss_op = tf.tensordot(yuv_blur_loss_op, yuv_scales, axes=1)

    # L_D loss
    D_loss = D_output_real - D_output_fake
    G_loss = D_output_fake

    loss_op = loss_scales[0] * secret_loss_op + loss_scales[1] * image_loss_op +loss_scales[2] * image_blur_loss_op

    if not args.no_gan:
        loss_op = loss_op + loss_scales[3]*G_loss

    return loss_op, secret_loss_op, image_loss_op, image_blur_loss_op, G_loss, D_loss, bit_acc


def prepare_deployment_hiding_graph(encoder, secret_input, image_input):

    residual = encoder((secret_input, image_input))
    encoded_image = image_input + residual
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    return encoded_image, residual


def prepare_deployment_reveal_graph(decoder, image_input):
    decoded_secret = decoder(image_input)

    return tf.round(tf.sigmoid(decoded_secret))
