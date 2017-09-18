#-------------------------------------------------------------------------
# Name:        caffe_ftr
# Purpose:
#
# Author:      wuhao
#
# Created:     14/07/2014
# Copyright:   (c) wuhao 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------

import os
import os.path as osp

import numpy as np
#import scipy.io as sio
import skimage.io

from collections import OrderedDict

import time

caffe_root = '/opt/caffe/'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


def load_image_list(img_dir, list_file_name):
    #list_file_path = os.path.join(img_dir, list_file_name)
    f = open(list_file_name, 'r')
    image_fullpath_list = []

    for line in f:
        if line.startswith('#'):
            continue

        items = line.split()
        image_fullpath_list.append(os.path.join(img_dir, items[0].strip()))

    f.close()

    return image_fullpath_list


def blobs_data(blob):
    try:
        d = blob.const_data
        # print 'GPU mode.'
    except AttributeError:
        # print 'GPU mode not support.'
        d = blob.data
    return d


def blobs_diff(blob):
    try:
        d = blob.const_diff
    except AttributeError:
        # print 'GPU mode not support.'
        d = blob.diff
    return d


def detect_GPU_extract_support(net):
    k, blob = net.blobs.items()[0]
    gpu_support = 0
    try:
        d = blob.const_data
        gpu_support = 1
    except AttributeError:
        gpu_support = 0
    return gpu_support


def extract_feature(network_proto_path,
                    network_model_path,
                    image_list,
                    data_mean,
                    layer_name,
                    image_as_grey=False):
    """
    Extracts features for given model and image list.

    Input
    network_proto_path: network definition file, in prototxt format.
    network_model_path: trainded network model file
    image_list: A list contains paths of all images, which will be fed into the
                network and their features would be saved.
    layer_name: The name of layer whose output would be extracted.
    save_path: The file path of extracted features to be saved.
    """
    # network_proto_path, network_model_path = network_path

    #--->added by zhaoyafei 2017-05-09
#    caffe.set_phase_test()
    if data_mean is not None and type(data_mean) is str:
        data_mean = np.load(data_mean)

    caffe.set_mode_gpu()
# net = caffe.Classifier(network_proto_path, network_model_path, None,
# data_mean, None, None, (2,1,0))
    net = caffe.Classifier(network_proto_path,
                           network_model_path,
                           None, data_mean,
                           0.0078125, 255, (2, 1, 0))
#    net = caffe.Classifier(network_proto_path, network_model_path, None, data_mean, 2.0, 1.0, (2,1,0))
   #--->end added by zhaoyafei 2017-05-09

    #--->commented by zhaoyafei 2017-05-09
#    net = caffe.Classifier(network_proto_path, network_model_path)
#    net.set_phase_test()
#
#    net.set_mode_gpu()
#
#    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#    #net.set_mean('data', caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')  # ImageNet mean
#    net.set_mean('data', data_mean)
#    if not image_as_grey:
#        net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
#
#    #net.set_input_scale('data', 256)  # the reference model operates on images in [0,255] range instead of [0,1]
#    net.set_input_scale('data', 1)
    #--->end commented by zhaoyafei 2017-05-09

    # img_list = [caffe.io.load_image(p) for p in image_file_list]

    #----- test

    blobs = OrderedDict([(k, v.data) for k, v in net.blobs.items()])

    # blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])
    input_shape = blobs['data'].shape
    batch_size = input_shape[0]
    print 'original input data shape: ', input_shape
    print 'original batch_size: ', batch_size

    shp = blobs[layer_name].shape
    print 'feature map shape: ', shp
    # print 'debug-------\nexit'
    # exit()

    # params = OrderedDict( [(k, (v[0].data,v[1].data)) for k, v in net.params.items()])
    # features_shape = (len(image_list), shp[1], shp[2], shp[3])
    # features_shape = (len(image_list), shp[1])
    features_shape = (len(image_list),) + shp[1:]
    features = np.empty(features_shape, dtype='float32', order='C')
    img_batch = []

    cnt_load_img = 0
    cnt_predict = 0

    time_load_img = 0.0
    time_predict = 0.0

    for cnt, path in zip(range(features_shape[0]), image_list):
        t1 = time.clock()
        img = caffe.io.load_image(path, color=not image_as_grey)
        if image_as_grey and img.shape[2] != 1:
            img = skimage.color.rgb2gray(img)
            img = img[:, :, np.newaxis]
        if cnt == 0:
            print 'image shape: ', img.shape
        # print img[0:10,0:10,:]
        # exit()
        img_batch.append(img)
        t2 = time.clock()

        cnt_load_img += 1
        time_predict += (t2 - t1)

        # print 'image shape: ', img.shape
        # print path, type(img), img.mean()
        if (len(img_batch) == batch_size) or cnt == features_shape[0] - 1:
            n_imgs = len(img_batch)
            t1 = time.clock()
            scores = net.predict(img_batch, oversample=False)
            t2 = time.clock()
            time_predict += (t2 - t1)
            cnt_predict += n_imgs

            '''
            print 'blobs[%s].shape' % (layer_name,)
            tmp =  blobs[layer_name]
            print tmp.shape, type(tmp)
            tmp2 = tmp.copy()
            print tmp2.shape, type(tmp2)
            print blobs[layer_name].copy().shape
            print cnt, n_imgs
            print batch_size
            # exit()

            # print img_batch[0:10]
            # print blobs[layer_name][:,:,0,0]
            # exit()
            '''

            # must call blobs_data(v) again, because it invokes (mutable_)cpu_data() which
            # syncs the memory between GPU and CPU
            blobs = OrderedDict([(k, v.data) for k, v in net.blobs.items()])

            print 'predict %d images cost %f seconds, average time: %f seconds' % (cnt_predict, time_predict, time_predict / cnt_predict)

            print '%d images processed' % (cnt + 1,)

            # print blobs[layer_name][0,:,:,:]
            # items of blobs are references, must make copy!
            # features[cnt-n_imgs+1:cnt+1, :,:,:] = blobs[layer_name][0:n_imgs,:,:,:].copy()
            # features[cnt-n_imgs+1:cnt+1, :] = blobs[layer_name][0:n_imgs,:].copy()
            ftrs = blobs[layer_name][0:n_imgs, ...]
            features[cnt - n_imgs + 1:cnt + 1, ...] = ftrs.copy()
            img_batch = []

        # features.append(blobs[layer_name][0,:,:,:].copy())

    print('Load %d images, cost %f seconds, average time: %f seconds' %
          (cnt_load_img, time_load_img, time_load_img / cnt_load_img))
    print('Predict %d images, cost %f seconds, average time: %f seconds' %
          (cnt_predict, time_predict, time_predict / cnt_predict))

    features = np.asarray(features, dtype='float32')
    return features


def extract_features_to_npy(network_proto_path,
                            network_model_path,
                            data_mean,
                            image_dir,
                            list_file,
                            layer_name,
                            save_path,
                            image_as_grey=False):

    img_list = load_image_list(image_dir, list_file)
    print img_list[0:10]
    # exit()


    ftrs = extract_feature(network_proto_path,
                           network_model_path,
                           img_list,
                           data_mean,
                           layer_name,
                           image_as_grey)


    if not osp.exists(save_path):
        os.makedirs(save_path)

    for i,fn in enumerate(img_list):
        base_name = osp.basename(fn)
        base_name = osp.splitext(base_name)[0]
        save_name = osp.join(save_path, base_name + '_feat.npy')
        np.save(save_name, ftrs[i])

    return


def print_usage():
    print 'To extract features:'
    print '  Extract features and save each feature into a .npy file.'
    print '  Usage: python caffe_ftr.py network_def trained_model mean_file image_dir image_list_file layer_name save_path'
    print '    network_def: network definition prototxt file'
    print '    trained_model: trained network model file, such as deep_iter_10000'
    print '    mean_file: If no mean file used, use -nomean as mean_file. mean_file should be numpy saved file (.npy).'
    print '    image_dir: the root dir of images'
    print '    image_list_file: a txt file, each line contains an image file path relative to image_dir and a label, seperated by space'
    print '    layer_name: name of the layer, whose outputs will be extracted'
    print '    save_path: path to save features'


def main(argv):

    if len(argv) < 7:
        print_usage()
        exit()

    if cmp(argv[3].lower(), '-nomean') == 0:
        argv[3] = None

    start_time = time.time()
    extract_features_to_npy(
        argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6])
    end_time = time.time()
    print 'time used: %f s\n' % (end_time - start_time,)


if __name__ == '__main__':
    if len(sys.argv)==1:
        ###centerface
#        network_def = r'C:\zyf\dnn_models\face_models\centerloss\center_face_deploy.prototxt'
#        trained_model = r'C:\zyf\dnn_models\face_models\centerloss\center_face_model.caffemodel'
#        mean_file = r'C:\zyf\dnn_models\face_models\centerloss\center_face_mean_127.5_1x3.npy'
#        layer_name = 'fc5'

        ###normface (not work, because 'Flip' layer is not defined in BVLC caffe)
#        network_def = r'C:\zyf\dnn_models\face_models\norm_face\Center_Face_99.2\face_deploy.prototxt'
#        trained_model = r'C:\zyf\dnn_models\face_models\norm_face\Center_Face_99.2\face_train_test_iter_6000.caffemodel'
#        mean_file = r'C:\zyf\dnn_models\face_models\centerloss\center_face_mean_127.5_1x3.npy'
#        layer_name = 'eltmax_fc5'

        ###sphereface
        network_def = r'C:\zyf\dnn_models\face_models\centerloss\center_face_deploy.prototxt'
#        trained_model = r'C:\zyf\dnn_models\face_models\\sphere_face_cwl\sphereface_iter_22000.caffemodel'
        trained_model = r'C:\zyf\dnn_models\face_models\\sphere_face\sphereface_model_iter_28000_bs512.caffemodel'
        mean_file = r'C:\zyf\dnn_models\face_models\centerloss\center_face_mean_127.5_1x3.npy'
        layer_name = 'fc5'

#        image_dir = r'./face_chips'
#        image_list_file = r'face_chips\face_chips_list.txt'
        image_dir = r'C:\zyf\github\mtcnn-caffe-good\face_aligner\face_chips'
        image_list_file = r'face_chips\face_chips_list_2.txt'
        save_path = 'extracted_features_2'
        argv = [network_def, trained_model, mean_file,
                image_dir, image_list_file,
                layer_name, save_path]
        main(argv)
    else:
        main(sys.argv[1:])

