import glob
from PIL import Image
import numpy as np
import os
import random
import tensorflow as tf
import utils
import models
import math
from os.path import join
import sys
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

slim = tf.contrib.slim

TRAIN_PATH = './data/images2'
CHECKPOINTS_PATH = './checkpoints/'
SAVED_MODELS = './saved_models'


if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)


def get_img_batch(files_list,
                  secret_size,
                  batch_size=4,
                  size=(400,400)):

    batch_cover = []
    batch_secret = []

    for i in range(batch_size):
        img_cover_path = random.choice(files_list)
        try:
            img_cover = Image.open(img_cover_path).convert("RGB")
            img_cover = img_cover.resize(size, Image.ANTIALIAS)
            img_cover = np.array(img_cover, dtype=np.float32) / 255.

        except:
            img_cover = np.zeros((size[0], size[1], 3), dtype=np.float32)

        batch_cover.append(img_cover)

        secret = np.random.binomial(1, .5, secret_size)
        batch_secret.append(secret)

    batch_cover, batch_secret = np.array(batch_cover), np.array(batch_secret)

    return batch_cover, batch_secret


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--secret_size', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=140000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--l2_loss_scale', type=float, default=4)  #4
    parser.add_argument('--l2_loss_ramp', type=int, default=15000)
    parser.add_argument('--secret_loss_scale', type=float, default=1.5)   #1.5
    parser.add_argument('--secret_loss_ramp', type=int, default=1.)
    parser.add_argument('--smoothing_loss_scale', type=float, default=1.5)   #1.5
    parser.add_argument('--smoothing_loss_ramp', type=int, default=15000)
    parser.add_argument('--G_loss_scale', type=float, default=.5)
    parser.add_argument('--G_loss_ramp', type=int, default=15000)
    parser.add_argument('--y_scale', type=float, default=1.0)
    parser.add_argument('--u_scale', type=float, default=10.0)
    parser.add_argument('--v_scale', type=float, default=10.0)
    parser.add_argument('--no_gan', action='store_true')
    parser.add_argument('--rnd_trans', type=float, default=.1) #0.06
    parser.add_argument('--rnd_bri', type=float, default=.3)
    parser.add_argument('--rnd_noise', type=float, default=.07) #0.05
    parser.add_argument('--rnd_sat', type=float, default=0.1)
    parser.add_argument('--rnd_hue', type=float, default=.1)  #0.05
    parser.add_argument('--contrast_low', type=float, default=0.5)
    parser.add_argument('--contrast_high', type=float, default=1.5)
    parser.add_argument('--jpeg_quality', type=float, default=50)
    parser.add_argument('--no_jpeg', action='store_true')
    parser.add_argument('--rnd_trans_ramp', type=int, default=15000)
    parser.add_argument('--rnd_bri_ramp', type=int, default=1000)
    parser.add_argument('--rnd_sat_ramp', type=int, default=1000)
    parser.add_argument('--rnd_hue_ramp', type=int, default=1000)
    parser.add_argument('--rnd_noise_ramp', type=int, default=1000)
    parser.add_argument('--contrast_ramp', type=int, default=1000)
    parser.add_argument('--jpeg_quality_ramp', type=float, default=1000)
    parser.add_argument('--no_im_loss_steps', help="Train without image loss for first x steps", type=int, default=5000)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--perceptual_model', type=str,
                        default="./saved_perceptual_model/save_new")
    args = parser.parse_args()

    files_list = glob.glob(join(TRAIN_PATH, "*"))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    height = 400
    width = 400
    secret_pl = tf.placeholder(shape=[None, args.secret_size], dtype=tf.float32, name="input_prep")
    image_pl = tf.placeholder(shape=[None, height, width, 3], dtype=tf.float32, name="input_hide")
    M_pl = tf.placeholder(shape=[None, 2, 8], dtype=tf.float32, name="input_transform")
    global_step_tensor = tf.Variable(6000, trainable=False, name='global_step')
    loss_scales_pl = tf.placeholder(shape=[4], dtype=tf.float32, name="input_loss_scales")
    yuv_scales_pl = tf.placeholder(shape=[3], dtype=tf.float32, name="input_yuv_scales")
    seg_pro_pl = tf.placeholder(shape=[None, height, width, 1], dtype=tf.float32, name="input_seg_probabilities")

    encoder = models.ChartStampEncoder()
    decoder = models.ChartStampDecoder(secret_size=args.secret_size)
    discriminator = models.Discriminator()

    # initialize perceptual model
    seg_graph = tf.Graph()
    with seg_graph.as_default():
        seg_sess = tf.Session()
        seg_model = tf.saved_model.loader.load(seg_sess, [tag_constants.SERVING], args.perceptual_model)

        seg_input_name = seg_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[
            'image_input'].name
        seg_input = seg_graph.get_tensor_by_name(seg_input_name)

        seg_output_name = seg_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
            'probabilities'].name
        seg_output = seg_graph.get_tensor_by_name(seg_output_name)

    # build ChartStamp model
    loss_op, secret_loss_op, image_loss_op, image_smoothing_loss_op, G_loss_op, D_loss_op, bit_acc_op= models.build_model(
            encoder=encoder,
            decoder=decoder,
            discriminator=discriminator,
            secret_input=secret_pl,
            image_input=image_pl,
            M=M_pl,
            loss_scales=loss_scales_pl,
            yuv_scales=yuv_scales_pl,
            seg_pro=seg_pro_pl,
            args=args,
            global_step=global_step_tensor)

    tvars=tf.trainable_variables()
    d_vars=[var for var in tvars if 'discriminator' in var.name]
    g_vars=[var for var in tvars if 'chart_stamp' in var.name]

    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

    learning_rate = tf.train.exponential_decay(args.lr,
                                               global_step=global_step_tensor,
                                               decay_steps=4000,
                                               decay_rate=0.95,
                                               )

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op, var_list=g_vars, global_step=global_step_tensor)
    train_secret_op = tf.train.AdamOptimizer(learning_rate).minimize(secret_loss_op, var_list=g_vars, global_step=global_step_tensor)

    optimizer = tf.train.RMSPropOptimizer(.00001)
    gvs = optimizer.compute_gradients(D_loss_op, var_list=d_vars)
    capped_gvs = [(tf.clip_by_value(grad, -.25, .25), var) for grad, var in gvs]
    train_dis_op = optimizer.apply_gradients(capped_gvs)

    deploy_hide_image_op, residual_op = models.prepare_deployment_hiding_graph(encoder, secret_pl, image_pl)
    deploy_decoder_op = models.prepare_deployment_reveal_graph(decoder, image_pl)

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=500, keep_checkpoint_every_n_hours=2)
    sess.run(tf.global_variables_initializer())

    if args.pretrained is not None:
        saver.restore(sess, args.pretrained)

    total_steps = len(files_list)//args.batch_size + 1
    global_step = 6000

    csvFile = open('./log_result.csv', 'a', newline='')
    writer = csv.writer(csvFile)
    data_column = ["global step", "total loss", "L2 loss", "secret loss", "smoothing loss",
                   "g loss", "bit acc", "lr", "secret_loss_scale"]

    writer.writerow(data_column)
    l2_loss_scale=args.l2_loss_scale
    secret_loss_scale = args.secret_loss_scale
    smoothing_loss_scale = args.smoothing_loss_scale
    G_loss_scale = args.G_loss_scale

    avg_acc = 0
    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps-global_step)):
            no_im_loss = global_step < args.no_im_loss_steps
            images, secrets = get_img_batch(files_list=files_list,
                                                     secret_size=args.secret_size,
                                                     batch_size=args.batch_size,
                                                     size=(height, width))

            # get perceptual importance map
            output_image = seg_sess.run(seg_output, feed_dict={seg_input: images})
            predictions = output_image[:, :, :, 1]
            segmentations = np.array(predictions).reshape([-1, 400, 400, 1])
            segmentations[segmentations > 0.9] = 1.0

            if global_step <= 40000:
                if not no_im_loss:
                    # adding l2/smoothing/g loss, loss scale increase from 0 to full range
                    l2_loss_scale = min(args.l2_loss_scale * (global_step - args.no_im_loss_steps) / args.l2_loss_ramp, args.l2_loss_scale)
                    secret_loss_scale = min(args.secret_loss_scale * (global_step - args.no_im_loss_steps) / args.secret_loss_ramp,
                                            args.secret_loss_scale)
                    smoothing_loss_scale = min(args.smoothing_loss_scale * (global_step - args.no_im_loss_steps) / args.smoothing_loss_ramp,
                                            args.smoothing_loss_scale)
                    G_loss_scale = min(args.G_loss_scale * (global_step - args.no_im_loss_steps) / args.G_loss_ramp,
                                       args.G_loss_scale)

            # get perspective trans matrix M
            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran
            M = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)

            feed_dict = {secret_pl: secrets,
                         image_pl: images,
                         M_pl: M,
                         loss_scales_pl:[secret_loss_scale, l2_loss_scale, smoothing_loss_scale, G_loss_scale],
                         yuv_scales_pl:[args.y_scale, args.u_scale, args.v_scale],
                         seg_pro_pl: segmentations,
                         }

            if no_im_loss:
                _, secert_ls, global_step, bit_acc, current_lr = sess.run([train_secret_op, secret_loss_op, global_step_tensor, bit_acc_op, learning_rate], feed_dict)
                print("secret loss:%.8f" % secert_ls, " bit acc:%2.4f" % bit_acc, " lr:", current_lr, " global step:", global_step)

            else:
                _, global_step, total_ls, secret_loss, l2_loss, image_smoothing_loss, g_loss, bit_acc, current_lr = \
                    sess.run([train_op, global_step_tensor, loss_op, secret_loss_op, image_loss_op,
                              image_smoothing_loss_op, G_loss_op, bit_acc_op, learning_rate], feed_dict)
                if not args.no_gan:
                    sess.run([train_dis_op, clip_D], feed_dict)

                # adaptive change secret loss scale according to avg acc on validation set
                avg_acc += bit_acc
                if global_step > 40000:
                    roun = 4000
                    if global_step % roun == roun-500:
                        avg_acc = 0
                    if global_step % roun == 0:
                        avg_acc /= 500
                        if avg_acc < 0.95:
                            secret_loss_scale *= 1.1
                        else:
                            if avg_acc > 0.955:
                                secret_loss_scale /= math.pow(1.1, 1.5)
                        secret_loss_scale = np.minimum(secret_loss_scale, 10)
                        secret_loss_scale = np.maximum(secret_loss_scale, 0.01)
                        print("avg acc:%2.4f, secret loss scale:" % avg_acc, secret_loss_scale, "\n")
                        avg_acc = 0

                # show training log
                print("loss:%2.8f" % total_ls, " secret loss:%2.8f" % (secret_loss * secret_loss_scale),
                      " L2 image loss:%2.8f" % (l2_loss * l2_loss_scale),
                      " image blur loss:%2.8f" % (image_smoothing_loss * smoothing_loss_scale),
                      " G_loss:%2.8f" % (g_loss*G_loss_scale), " bit acc:%2.4f" % bit_acc,
                      "	global step:", global_step, "  lr:", current_lr, " secret scale:", secret_loss_scale)
                data_result = [global_step]+[total_ls]+[secret_loss * secret_loss_scale]+\
                              [l2_loss * l2_loss_scale]+[image_smoothing_loss * smoothing_loss_scale]+\
                              [g_loss*G_loss_scale]+[bit_acc]+[current_lr]+[secret_loss_scale]
                writer.writerow(data_result)

            if global_step % 5000 == 0:
                saver.save(sess, "./checkpoint/model.ckpt", global_step=global_step)

    saver.save(sess, "./checkpoint/model.ckpt", global_step=global_step)

    # freeze model
    constant_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            [deploy_hide_image_op.name[:-2], residual_op.name[:-2], deploy_decoder_op.name[:-2]])
    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(constant_graph_def, name='')
        tf.saved_model.simple_save(session,
                                   SAVED_MODELS + '/' + 'model_' + str(args.secret_size),
                                   inputs={'secret':secret_pl, 'image':image_pl},
                                   outputs={'stegastamp':deploy_hide_image_op, 'residual':residual_op, 'decoded':deploy_decoder_op})

    writer.close()

if __name__ == "__main__":
    main()
