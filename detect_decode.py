import glob
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import bchlib
import sys
import torch
import torch.nn as nn
import cv2
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

slim = tf.contrib.slim
sys.path.append(r"./detect_model")
from model import East
from eval import predict_one

os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


BCH_POLYNOMIAL = 137
BCH_BITS = 5


def get_accuary(secret,recover):
    str_length = len(secret)
    acc_bit=0.0
    for i in range(str_length):
        if secret[i] == recover[i]:
            acc_bit = acc_bit+1.
    return acc_bit/str_length


def load_EAST_model(checkpoint_path):
    model = East()
    model = model.cuda()
    model = nn.DataParallel(model).cuda()
    if os.path.isfile(checkpoint_path):
        weightpath = os.path.abspath(checkpoint_path)
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Begin".format(weightpath))
        checkpoint = torch.load(weightpath)
        model.load_state_dict(checkpoint['state_dict'])
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Done".format(weightpath))
    print('EAST <==> Prepare <==> Network <==> Done')
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./saved_models/save_100')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default='./100bit_Display_Captured')
    parser.add_argument('--detect_model_path', type=str, default='./detect_model/weight_noise7/epoch_75_checkpoint.pth')
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

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    EAST_model = load_EAST_model(args.detect_model_path)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    for filename in files_list:
        box = predict_one(EAST_model, filename)
        ori_img = cv2.imread(filename)
        blur_ori_img = cv2.GaussianBlur(ori_img, (3, 3), 0)
        frame_rgb = cv2.cvtColor(blur_ori_img, cv2.COLOR_BGR2RGB)
        pts_dst = np.array([[0, 0], [399, 0], [399, 399], [0, 399]])
        h, status = cv2.findHomography(box, pts_dst)

        try:
            warped_im = cv2.warpPerspective(frame_rgb, h, (400, 400), flags=cv2.INTER_AREA)
            warped_im = warped_im.astype(np.uint8)
            im = Image.fromarray(np.array(warped_im))
            im.save('warped.png')
            w_im = warped_im.astype(np.float32)
            w_im /= 255.
        except:
            print("picture:", filename, "detect failed!")
            continue
        for im_rotation in range(1):
            w_rotated = np.rot90(w_im, im_rotation)
            decode_b = sess.run([output_secret], feed_dict={input_image: [w_rotated]})[0][0]
            recover_secret = [int(x) for x in decode_b]
            # acc = get_accuary(secret, recover_secret)
            # print("accuary:", acc)
            packet_binary = "".join([str(bit) for bit in recover_secret[:96]])
            packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
            packet = bytearray(packet)

            data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

            bitflips = bch.decode_inplace(data, ecc)

            if bitflips != -1:
                print('Num bits corrected: ', bitflips)
                try:
                    code = data.decode("utf-8")
                    print(code)
                except:
                    continue






if __name__ == "__main__":
    main()
