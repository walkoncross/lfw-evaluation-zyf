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

from collections import OrderedDict
import gzip
import zipfile
import cPickle
import time

import numpy as np
import scipy.io as sio
import skimage.io
import os
import os.path as osp

import time

caffe_root = '/opt/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

gpu_id = 0

PLUS_CONCAT = True
# MIRROR_COMBINE_METHOD = 'elt-max'
# MIRROR_COMBINE_METHOD = 'elt-avg'

NO_INPUT_SCALE = False

if not NO_INPUT_SCALE:
	# for centerface/normface/sphereface models
	raw_scale = 255
	input_scale = 0.0078125
else:
	# for vggface/face-resnet (other face models finetuned from imagenet models)
	raw_scale = 255
	input_scale = 1.0


class UnpickleError(Exception):
    pass


def pickle(filename, data, compress=False):
    if compress:
        fo = zipfile.ZipFile(
            filename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        fo.writestr('data', cPickle.dumps(data, -1))
    else:
        fo = open(filename, "wb")
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()


def unpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)
    f = open(filename, 'rb')
    header = f.read(4)
    f.close()
    if cmp(header, '\x50\x4b\x03\x04') == 0:
        fo = zipfile.ZipFile(filename, 'r', zipfile.ZIP_DEFLATED)
        dict = cPickle.loads(fo.read('data'))
    else:
        fo = open(filename, 'rb')
        dict = cPickle.load(fo)
    fo.close()
    return dict


def load_image_list(img_dir, list_file_name):
    #list_file_path = os.path.join(img_dir, list_file_name)
    f = open(list_file_name, 'r')
    image_fullpath_list = []
    image_list = []
    labels = []
    for line in f:
        items = line.split()
        image_list.append(items[0].strip())
        image_fullpath_list.append(os.path.join(img_dir, items[0].strip()))
        labels.append(items[1].strip())
    return image_fullpath_list, labels, image_list


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
    caffe.set_device(gpu_id)
# net = caffe.Classifier(network_proto_path, network_model_path, None,
# data_mean, None, None, (2,1,0))
    net = caffe.Classifier(network_proto_path,
                           network_model_path,
                           None, data_mean,
                           input_scale, raw_scale,
                           (2, 1, 0))
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

    if PLUS_CONCAT:
        if batch_size < 2:
            raise Exception('batch_size of input "data" layer must > 1')

        batch_size /= 2
        # new_input_shape = (input_shape[0]/2, ) + input_shape[1:]
        # blobs['data'].reshape(new_input_shape)
        # print 'reshape input data into for mirror eval: ',
        # blobs['data'].shape
        print 'halve the batch_size for mirror eval: batch_size=', batch_size

    shp = blobs[layer_name].shape
    print 'feature map shape: ', shp
    # print 'debug-------\nexit'
    # exit()

    # params = OrderedDict( [(k, (v[0].data,v[1].data)) for k, v in net.params.items()])
    # features_shape = (len(image_list), shp[1], shp[2], shp[3])
    # features_shape = (len(image_list), shp[1])
    features_shape = (len(image_list),) + shp[1:]
    features = np.empty(features_shape, dtype='float32', order='C')
    if PLUS_CONCAT:
        features_concat = np.empty(
            (len(image_list), np.array(shp[1:]).prod()*2), dtype='float32', order='C')

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
            if PLUS_CONCAT:
                for i in range(n_imgs):
                    mirror_img = np.fliplr(img_batch[i])
                    img_batch.append(mirror_img)

                print 'add mirrored images into predict batch'
                print 'after add: len(img_batch)=%d' % (len(img_batch))

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

            print 'predict %d images cost %f seconds, average time: %f seconds' % (cnt_predict,
                                                                                   time_predict, time_predict / cnt_predict)

            print '%d images processed' % (cnt + 1,)

            # print blobs[layer_name][0,:,:,:]
            # items of blobs are references, must make copy!
            # features[cnt-n_imgs+1:cnt+1, :,:,:] = blobs[layer_name][0:n_imgs,:,:,:].copy()
            # features[cnt-n_imgs+1:cnt+1, :] = blobs[layer_name][0:n_imgs,:].copy()
            # ftrs = blobs[layer_name]

            if PLUS_CONCAT:
                ftrs = blobs[layer_name][0:n_imgs * 2, ...]

                concat_ftrs = np.hstack((ftrs[:n_imgs], ftrs[n_imgs:]))

                features[cnt - n_imgs + 1:cnt +
                         1, ...] = ftrs[:n_imgs, ...].copy()
                features_concat[cnt - n_imgs +
                                       1:cnt + 1, ...] = concat_ftrs
            else:
                # features[cnt-n_imgs+1:cnt+1, ...] = blobs[layer_name][0:n_imgs, ...].copy()
                ftrs = blobs[layer_name][0:n_imgs, ...]
                features[cnt - n_imgs + 1:cnt + 1, ...] = ftrs.copy()

            img_batch = []

        # features.append(blobs[layer_name][0,:,:,:].copy())

    print('Load %d images, cost %f seconds, average time: %f seconds' %
          (cnt_load_img, time_load_img, time_load_img / cnt_load_img))
    print('Predict %d images, cost %f seconds, average time: %f seconds' %
          (cnt_predict, time_predict, time_predict / cnt_predict))

    features = np.asarray(features, dtype='float32')
    if PLUS_CONCAT:
        features_concat = np.asarray(
            features_concat, dtype='float32')
        return [features, features_concat]
    else:
        return [features]


def extract_features_to_mat(network_proto_path,
                            network_model_path,
                            data_mean,
                            image_dir,
                            list_file,
                            layer_name,
                            save_path,
                            image_as_grey=False):
    img_list, labels, img_list_original = load_image_list(image_dir, list_file)
    print img_list[0:10]
    print labels[0:10]
    # exit()

    float_labels = labels_list_to_float(labels)

    ftrs = extract_feature(network_proto_path,
                           network_model_path,
                           img_list,
                           data_mean,
                           layer_name,
                           image_as_grey)
    # print ftr.shape
#    if ftr.shape[3]==1 and ftr.shape[2]==1:
#        ftr = ftr[:,:,0,0]
    # print ftr.shape
    # labels = np.asarray(labels, dtype='float32')
    float_labels = labels_list_to_float(labels)

    dic = {'features': ftrs[0],
           'labels': float_labels,
           'labels_original': string_list_to_cells(labels),
           'image_path': string_list_to_cells(img_list_original)}
    sio.savemat(save_path, dic)

    if len(ftrs) > 1:
        splits = osp.splitext(save_path)
        save_path2 = splits[0] + '_concat' + splits[1]

        dic['features'] = ftrs[1]
        sio.savemat(save_path2, dic)
    return


def string_list_to_cells(lst):
    """
    Uses numpy.ndarray with dtype=object. When save to mat file using scipy.io.savemat, it will be a cell array.
    """
    cells = np.ndarray(len(lst), dtype='object')
    for i in range(len(lst)):
        cells[i] = lst[i]
    return cells


def labels_list_to_float(labels):

    int_labels = []
    for e in labels:
        try:
            inte = int(e)
        except ValueError:
            print 'Labels are not int numbers. A mapping will be used.'
            break
        int_labels.append(inte)
    if len(int_labels) == len(labels):
        return int_labels

    labels_unique = list(sorted(set(labels)))
    print labels[0:10]
    print labels_unique[0:10]

    dic = dict([(lb, i)
                for i, lb in zip(range(len(labels_unique)), labels_unique)])
    labels_float = [dic[a] for a in labels]
    '''
    print labels
    print dic
    print labels_float
    '''
    return labels_float


def save_filters(network_def, network_model, save_path):
    # print 'arg1', network_def
    # print 'arg2', network_model
    # print 'arg3', save_path

    #--->added by zhaoyafei 2017-05-09
    caffe.set_phase_test()
    caffe.set_mode_cpu()
    #--->end added by zhaoyafei 2017-05-09

    net = caffe.Classifier(network_def, network_model)
    #--->commented by zhaoyafei 2017-05-09
#    net.set_phase_test()
#    net.set_mode_cpu()
    #--->end commented by zhaoyafei 2017-05-09

    '''
    net.set_mean('data', None)
    net.set_channel_swap('data', (2,1,0))
    net.set_input_scale('data', 256)

    data_shape = net.blobs['data'].data.shape[1:]
    print data_shape
    dummy_data = np.zeros(data_shape, dtype='float32')
    scores = net.predict([dummy_data], oversample=False)
    blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])
    '''
    params = []
    for k, v in net.params.items():
        print k, type(v), len(v)

        vlist = [vt.data for vt in v]
        params.append((k, vlist))

    # exit()
    # params = [(k, v) for k, v in net.params.items()]
    dc = dict(params)
    sio.savemat(save_path, dc)

    return


def save_features(network_def, network_model, mean_file, img_path, save_path):

    print img_path
    print 'hello'
    img = caffe.io.load_image(img_path)

   #--->added by zhaoyafei 2017-05-09
    data_mean = None
    if mean_file is not None:
        data_mean = np.load(mean_file)
    caffe.set_mode_cpu()
	caffe.set_device(gpu_id)
#    caffe.set_device(2)
#    net = caffe.Classifier(network_def, network_model, None, data_mean, None, None, (2,1,0))
    net = caffe.Classifier(network_def, network_model,
                           None, data_mean,
                           input_scale, raw_scale,
                           (2, 1, 0))
#    net = caffe.Classifier(network_def, network_model, None, data_mean, 2.0, 1.0, (2,1,0))
    #--->end added by zhaoyafei 2017-05-09

    #--->commented by zhaoyafei 2017-05-09
#    net = caffe.Classifier(network_def, network_model)
#    net.set_phase_test()
#    net.set_mode_cpu()
#    net.set_device(2)
#    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#    #net.set_mean('data', caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')  # ImageNet mean
#    net.set_mean('data', mean_file)
#    net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
#
#    #net.set_input_scale('data', 256)  # the reference model operates on images in [0,255] range instead of [0,1]
#    net.set_input_scale('data', 1)
    #--->commented by zhaoyafei 2017-05-09

    scores = net.predict([img], oversample=False)

    blobs = OrderedDict([(k, v.data) for k, v in net.blobs.items()])
    sio.savemat(save_path, blobs)
    return


def main(argv):

    # print argv[0]
    # print argv[0].lower()
    if len(argv) == 0:
        print 'To extract features:'
        print '  Extracts features and saves to mat file.'
        print '  Usage: python caffe_ftr.py network_def trained_model image_dir image_list_file layer_name save_file'
        print '    network_def: network definition prototxt file'
        print '    trained_model: trained network model file, such as deep_iter_10000'
        print '    image_dir: the root dir of images'
        print '    image_list_file: a txt file, each line contains an image file path relative to image_dir and a label, seperated by space'
        print '    layer_name: name of the layer, whose outputs will be extracted'
        print '    save_file: the file path to save features, better to ends with .mat'
        print 'To save filters:'
        print '  Saves filters to mat files.'
        print '  Usage: python caffe_ftr.py --save-filters network_def network_model save_path'
        print '    (args are similar.)'

        exit()

    cmd_str = argv[0].lower()

    if not cmd_str.startswith('--'):
        # old version
        if len(argv) != 6:
            print ' Extracts features and saves to mat file.'
            print ' Usage: python caffe_ftr.py network_def trained_model image_dir image_list_file layer_name save_file'
            print '    network_def: network definition prototxt file'
            print '    trained_model: trained network model file, such as deep_iter_10000'
            print '    image_dir: the root dir of images'
            print '    image_list_file: a txt file, each line contains an image file path relative to image_dir and a label, seperated by space'
            print '    layer_name: name of the layer, whose outputs will be extracted'
            print '    save_file: the file path to save features, better to ends with .mat'
            exit()
        start_time = time.time()
        extract_features_to_mat(
            argv[0], argv[1], None, argv[2], argv[3], argv[4], argv[5])
        end_time = time.time()
        print 'time used: %f s\n' % (end_time - start_time,)
        exit()

    # ---
    # new version
    # ---

    if cmp(cmd_str, '--save-filters') == 0:
        print 'command: save-filters'
        if len(argv) != 4:
            print '  Saves filters to mat files.'
            print '  Usage: python caffe_ftr.py --save-filters network_def network_model save_path'
            print '    (args are similar.)'
            exit()
        save_filters(argv[1], argv[2], argv[3])

    #-----save-features
    elif cmp(cmd_str, '--save-features') == 0:
        print 'command: save-features'
        if len(argv) != 6:
            print 'Given an image, saves all features (all layer outputs) to mat file.'
            print 'Usage: python caffe_ftr.py --save-features network_def network_model mean_file img_path save_path'
            print '  If no mean file used, use -nomean as mean_file. mean_file should be numpy saved file (.npy).'
            exit()
        if cmp(argv[3].lower(), '-nomean') == 0:
            save_features(argv[1], argv[2], None, argv[4], argv[5])
        else:
            save_features(argv[1], argv[2], argv[3], argv[4], argv[5])

    #------extract-features
    elif cmp(cmd_str, '--extract-features') == 0:
        print 'command: extract-features'
        if (len(argv) != 8) and (len(argv) != 9):
            print 'Given image list file and trained model, extract features and saves to mat file.'
            print '  Usage: python caffe_ftr.py --extract-features network_def trained_model mean_file image_dir image_list_file layer_name save_file [as_grey]'
            print '    If no mean file used, use -nomean as mean_file. mean_file should be numpy saved file (.npy).'
            print '    If as_grey = 1, images will be loaded as grey scale.'
            exit()
        if cmp(argv[3].lower(), '-nomean') == 0:
            argv[3] = None
        if len(argv) == 9:
            argv[-1] = (int(argv[-1]) == 1)
        start_time = time.time()
        extract_features_to_mat(*argv[1:])
        end_time = time.time()
        print 'time used: %f s\n' % (end_time - start_time,)
    else:
        print 'Unknown command: %s' % (cmd_str,)

    # -----------------
    # -----------------
    '''
    elif len(argv) != 6:
        print ' Extracts features and saves to mat file.'
        print ' Usage: python caffe_ftr.py network_def trained_model image_dir image_list_file layer_name save_file'
        print '    network_def: network definition prototxt file'
        print '    trained_model: trained network model file, such as deep_iter_10000'
        print '    image_dir: the root dir of images'
        print '    image_list_file: a txt file, each line contains an image file path relative to image_dir and a label, seperated by space'
        print '    layer_name: name of the layer, whose outputs will be extracted'
        print '    save_file: the file path to save features, better to ends with .mat'
    else:
        start_time = time.time()
        extract_features_to_mat(*argv)
        end_time = time.time()
        print 'time used: %f s\n' % (end_time - start_time,)
    '''
    return


if __name__ == '__main__':
    # print  'main'
    # print sys.argv
    '''
    lbs = ['ad','dd','ewrer','sdfd', 'aaa']
    lbs = ['0', '1', '4', '56', '2']
    lbs_float = labels_list_to_float(lbs)
    print lbs
    print lbs_float
    exit()
    '''
    main(sys.argv[1:])
