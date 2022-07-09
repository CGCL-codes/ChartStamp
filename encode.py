import bchlib
import glob
import os
from PIL import Image
import numpy as np
import tensorflow as tf

from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

BCH_POLYNOMIAL = 137
BCH_BITS = 5


def get_accuary(secret,recover):
    str_length = len(secret)
    acc_bit=0.0
    for i in range(str_length):
        if secret[i] == recover[i]:
            acc_bit = acc_bit+1.
    return acc_bit/str_length


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./saved_models/save_100')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default='../data/test_images')
    parser.add_argument('--save_dir', type=str, default='./embed_bit_100')
    parser.add_argument('--secret', type=str, default='chart!')
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_chartstamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_chartstamp = tf.get_default_graph().get_tensor_by_name(output_chartstamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    input_image_name_decode = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image_decode = tf.get_default_graph().get_tensor_by_name(input_image_name_decode)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    width = 400
    height = 400

    # encode input 56 bits
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    if len(args.secret) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return

    data = bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0,0,0,0])

    # only for debug, generate random # bits
    # secret = np.random.binomial(1, .5, 1000)

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        size = (width, height)
        for filename in files_list:
            image_ori = Image.open(filename).convert("RGB")
            size_img = image_ori.size
            image = np.array(image_ori.resize(size, Image.ANTIALIAS), dtype=np.float32)
            image /= 255.

            feed_dict = {input_secret:[secret],
                         input_image:[image]}

            hidden_img, residual = sess.run([output_chartstamp, output_residual],feed_dict=feed_dict)

            residual = ((residual[0] + 0.5) * 255).astype(np.uint8)
            im_residual = Image.fromarray(np.array(residual))
            im_residual = im_residual.resize(size_img, Image.BICUBIC)

            # resized residue + input image
            resized_residual = np.array(im_residual).astype(np.int)- 0.5 * 255
            rescaled = np.clip(resized_residual + np.array(image_ori).astype(np.int), 0, 255)
            rescaled = rescaled.astype(np.uint8)

            save_name = filename.split('/')[-1].split('.')[0]

            im = Image.fromarray(np.array(rescaled))
            str1 = args.save_dir + '/encoded_img/'+save_name+'_embedded.png'
            im.save(str1)

            im_residual.save(args.save_dir + '/residue/'+save_name+'_residual.png')

            image = Image.open(str1).convert("RGB")
            image = np.array(image.resize((400, 400), Image.ANTIALIAS), dtype=np.float32)
            image /= 255.

            decode_b = sess.run([output_secret], feed_dict={input_image_decode: [image]})[0][0]
            recover_secret = [int(x) for x in decode_b]
            acc = get_accuary(secret, recover_secret)
            print(acc)


if __name__ == "__main__":
    main()
