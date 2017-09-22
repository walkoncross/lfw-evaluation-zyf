"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are pre-calculated and saved into a matlab .mat file.
"""

import numpy as np
import argparse
import sys
import os
import os.path as osp

from load_features import load_mat_features
from draw_pr_curve_zyf import calc_roc, calc_presicion_recall, draw_analysis_figure


def load_image_list(list_file, load_lines=None):
    print "Load image list form file: ", list_file
    fp = open(list_file, 'r')
    img_list = []
    idx_list = []
    id_list = []

    cnt = 0

    for line in fp:
        line_split = line.split()
        img_list.append(line_split[0])
        if len(line_split) > 2:  # for line format: <img_name> <img_idx> <img_id>
            idx_list.append(int(line_split[1]))
            id_list.append(int(line_split[2]))
        else:  # for line format: <img_name> <img_id>
            idx_list.append(cnt)
            id_list.append(int(line_split[1]))

        cnt += 1
        if load_lines > 0 and cnt == load_lines:
            break

    fp.close()

    return (img_list, np.array(idx_list), np.array(id_list))


def generate_gt_mat(probe_id_list, gallery_id_list):
    num_id1 = len(probe_id_list)
    num_id2 = len(gallery_id_list)

    id_mat1 = np.array(probe_id_list)
    id_mat2 = np.array(gallery_id_list)
    gt_mat = np.zeros((num_id1, num_id2), dtype=np.ubyte)

    for i in range(num_id1):
        gt_mat[i] = (id_mat2 == id_mat1[i])

    return gt_mat


def calc_similarity_mat(probe_ftrs, gallery_ftrs, distance_type='cosine'):
    print "Calc similarities of pairs"

    if distance_type == 'squared':
        raise Exception('Only support cosine similarity right now')
        pass
    else:
        sim_mat = np.dot(probe_ftrs, gallery_ftrs.T)

    return sim_mat

# def calc_roc_local(similarity_mat, gt_mat, threshs):
#    num_threshs = len(threshs)
#
#    tp = np.zeros(num_threshs)
#    fn = np.zeros(num_threshs)
#    tn = np.zeros(num_threshs)
#    fp = np.zeros(num_threshs)
#
#    gt_mat = gt_mat.astype(np.ubyte)
#    gt_mat_neg = 1 - gt_mat
#
#    pos_pairs = gt_mat.sum()
#    neg_pairs = gt_mat.size() - pos_pairs
#
#    print "pos_pairs: ", pos_pairs
#    print "neg_pairs: ", neg_pairs
#
#    for (i, thr) in enumerate(threshs):
#        threshed_mat = (similarity_mat>thr).astype(np.ubyte)
#        t_tp = np.logical_and(threshed_mat, gt_mat).sum()
#        t_tn = np.logical_and(1 - threshed_mat, gt_mat_neg).sum()
#        t_fn = pos_pairs - t_tp
#        t_fp = neg_pairs - t_tn
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
    ftr_mat = load_mat_features(args.feature_mat_file, do_norm)

    gallery_img_list, gallery_idx_list,  gallery_id_list = load_image_list(
        args.gallery_list_file)
    probe_img_list, probe_idx_list, probe_id_list = load_image_list(
        args.probe_list_file)

#    gallery_img_list, gallery_idx_list,  gallery_id_list = load_image_list(
#        args.gallery_list_file, 100)
#    probe_img_list, probe_idx_list, probe_id_list = load_image_list(
#        args.probe_list_file, 100)

    gallery_ftrs = ftr_mat[np.array(gallery_idx_list)]
    print "gallery_ftrs.shape: ", gallery_ftrs.shape

    probe_ftrs = ftr_mat[np.array(probe_idx_list)]
    print "probe_ftrs.shape: ", probe_ftrs.shape

    n_gallery = len(gallery_img_list)
    n_probe = len(probe_img_list)

    gt_mat = generate_gt_mat(probe_id_list, gallery_id_list)
#    similarity_mat = np.dot(probe_ftrs, gallery_ftrs.T)
    similarity_mat = calc_similarity_mat(probe_ftrs, gallery_ftrs)
    print similarity_mat

    pos_pairs = gt_mat.sum()
    neg_pairs = gt_mat.size - pos_pairs

    fp_log.write('Total pairs: {}\n'.format(n_gallery * n_probe))
    fp_log.write('Effective pos pairs: {}\n'.format(pos_pairs))
    fp_log.write('Effective neg pairs: {}\n'.format(neg_pairs))

    fp_log.close()

    threshs = None
#    calc_roc_local(similarity_mat, gt_mat, threshs)
    tp, fn, tn, fp = calc_roc(similarity_mat, gt_mat, threshs, save_dir)

    calc_presicion_recall(tp, fn, tn, fp, threshs, save_dir)
    draw_analysis_figure(tp, fn, tn, fp, save_dir, True)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('feature_mat_file', type=str,
                        help='Path to the .mat feature file.')
    parser.add_argument('--gallery_list_file', type=str,
                        help='The file of gallery list',
                        default='../lfw_data//test_ident_gallery.txt')
    parser.add_argument('--probe_list_file', type=str,
                        help='The file of probe list',
                        default='../lfw_data//test_ident_probe.txt')
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

        gallery_list_file = '../lfw_data/test_ident_gallery.txt'
        probe_list_file = '../lfw_data/test_ident_probe.txt'
        save_dir = './eval_rlt'
        distance = 'cosine'

        sys.argv.append(ftr_path)
        sys.argv.append('--gallery_list_file={}'.format(gallery_list_file))
        sys.argv.append('--probe_list_file={}'.format(probe_list_file))
        sys.argv.append('--save_dir={}'.format(save_dir))
        sys.argv.append('--distance=' + distance)

    eval_roc_and_pr(sys.argv[1:])
