"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are pre-calculated and saved into a matlab .mat file.
"""

import numpy as np
import argparse
import sys
import os
import os.path as osp

from load_features import load_mat_features
from draw_allpairs_pr_curve_zyf import calc_roc, calc_presicion_recall, draw_analysis_figure


def load_image_list(list_file):
    fp = open(list_file, 'r')

    img_list = []
    id_list = []

    for line in fp:
        line = line.strip()
        if not line:
            continue

        line_split = line.split()

        img_name = line_split[0]
        img_id = int(line_split[1])

        img_list.append(img_name)
        id_list.append(img_id)

    fp.close()

    return (img_list, id_list)


def generate_gt_mat(id_list):
    num_ids = len(id_list)

    id_mat = np.array(id_list)
    gt_mat = np.zeros((num_ids, num_ids), dtype=np.ubyte)

    for i in range(num_ids):
        gt_mat[i] = (id_mat == id_mat[i])

    return gt_mat


def calc_similarity_mat(all_ftr_mat, distance_type='cosine'):
    print "Calc similarities of pairs"

    if distance_type == 'squared':
        raise Exception('Only support cosine similarity right now')
        pass
    else:
        sim_mat = np.dot(all_ftr_mat, all_ftr_mat.T)

    return sim_mat

# def calc_roc_local(similarity_mat, gt_mat, threshs):
#    mat_shape = gt_mat.shape
#    print 'mat_shape: ', mat_shape
#
#    n_thresh = len(threshs)
#
#    tp = np.zeros((n_thresh, ))
#    fn = np.zeros((n_thresh, ))
#    tn = np.zeros((n_thresh, ))
#    fp = np.zeros((n_thresh, ))
#
#    gt_mat = gt_mat.astype(np.ubyte)
#    gt_mat_neg = 1 - gt_mat
#
#    pos_all = gt_mat.sum()
#    neg_all_x2 = mat_shape[0]*mat_shape[1] - pos_all
#    pos_all_x2 = pos_all - mat_shape [0]
#
#    print "pos_all_x2: ", pos_all_x2
#    print "neg_all_x2: ", neg_all_x2
#
#    for (i, thr) in enumerate(threshs):
#        threshed_mat = (similarity_mat>thr).astype(np.ubyte)
#        t_tp = np.logical_and(threshed_mat, gt_mat).sum() - mat_shape[0]
#        t_tn = np.logical_and(1 - threshed_mat, gt_mat_neg).sum()
#        t_fn = pos_all_x2 - t_tp
#        t_fp = neg_all_x2 - t_tn
#
#        tp[i] = t_tp
#        tn[i] = t_tn
#        fn[i] = t_fn
#        fp[i] = t_fp
#
#    return tp, fn, tn, fp


def eval_roc_and_pr(argv):

    args = parse_arguments(argv)

    print 'args:', args
#    for k,v in args:
#        print "{}: {}".format(k,v)
    save_dir = args.save_dir

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_log = open(osp.join(save_dir, 'eval.log'), 'w')
    fp_log.write('args:\n{}\n\n'.format(args))

    do_norm = not(args.no_normalize)
    img_list, id_list = load_image_list(args.image_list_file)

#    ftr_mat = load_mat_features(args.feature_mat_file, do_norm)[:100]
#    gt_mat = generate_gt_mat(id_list[:100])

    ftr_mat = load_mat_features(args.feature_mat_file, do_norm)
    gt_mat = generate_gt_mat(id_list)

    num_imgs = len(img_list)
    pos_pairs = gt_mat.sum()
    neg_pairs = num_imgs * num_imgs - pos_pairs
    pos_pairs -= num_imgs
    fp_log.write('Effective pos pairs: {}\n'.format(pos_pairs*0.5))
    fp_log.write('Effective neg pairs: {}\n'.format(neg_pairs*0.5))
    fp_log.close()


    similarity_mat = calc_similarity_mat(ftr_mat)
    print similarity_mat

    threshs = None
#    calc_roc_local(similarity_mat, gt_mat, threshs)
    tp, fn, tn, fp = calc_roc(similarity_mat, gt_mat, threshs, save_dir)

    calc_presicion_recall(tp, fn, tn, fp, threshs, save_dir)
    draw_analysis_figure(tp, fn, tn, fp, save_dir, True)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('feature_mat_file', type=str,
                        help='Path to the .mat feature file.')
    parser.add_argument('--image_list_file', type=str,
                        help='The file of image list with person ids',
                        default='../lfw_data/lfw_list_mtcnn.txt')
    parser.add_argument('--save_dir', type=str,
                        help='directory for saving result files',
                        default='./eval_rlt')
    parser.add_argument('--distance', type=str,
                        help='how to calc distance: <cosine or squared>.',
                        default='cosine')
    parser.add_argument('--no_normalize', action='store_true',
                        help='do not normalize when loading features')
    return parser.parse_args(argv)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        #        ftr_path = 'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/LFW-mtcnn-aligned-96x112_center_face_model_orig.mat'
        ftr_path = 'C:/zyf/dnn_models/face_models/lfw_eval_results/LFW-mtcnn-simaligned-96x112_center_face_model_orig.mat'
#        ftr_path = r'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/LFW-mtcnn-simaligned-96x112_center_face_model_orig.mat'
        image_list_file = '../lfw_data/lfw_list_mtcnn.txt'

        save_dir = './eval_rlt'
        distance = 'cosine'

        sys.argv.append(ftr_path)
        sys.argv.append('--image_list_file={}'.format(image_list_file))
        sys.argv.append('--save_dir={}'.format(save_dir))
        sys.argv.append('--distance=' + distance)

    eval_roc_and_pr(sys.argv[1:])
