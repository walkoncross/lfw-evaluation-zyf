"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are pre-calculated and saved into a matlab .mat file.
"""

import numpy as np
import argparse
import sys
import os
import os.path as osp

sys.path.append('../io')

from load_features import load_mat_features
from draw_pr_curve_zyf import calc_roc, calc_presicion_recall, draw_analysis_figure


def load_image_pairs(pairs_file):
    print "Load image pairs form file: ", pairs_file
    fp = open(pairs_file, 'r')
    idx_list = []
    img_pair_list = []

    for line in fp:
        line_split = line.split()
        idx_list.append((int(line_split[0]), int(line_split[1])))
        img_pair_list.append((line_split[-1], line_split[-1]))

    fp.close()

    return (idx_list, img_pair_list)


def calc_similarity(all_ftr_mat, pairs_idx_list, distance_type='cosine'):
    print "Calc similarities of pairs"

    sim_list = []
    if distance_type == 'squared':
        for idx_pair in pairs_idx_list:
            dist_vec = all_ftr_mat[idx_pair[0]] - all_ftr_mat[idx_pair[1]]
            sim = -np.dot(dist_vec, dist_vec.T)
            sim_list.append(sim)
    else:
        for idx_pair in pairs_idx_list:
            sim = np.dot(all_ftr_mat[idx_pair[0]], all_ftr_mat[idx_pair[1]])
            sim_list.append(sim)

    return sim_list


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
    fp_log.close()

    do_norm = not(args.no_normalize)
    ftr_mat = load_mat_features(args.feature_mat_file, do_norm)

    same_pairs_idx_list, same_img_list = load_image_pairs(args.same_pairs_file)
    same_sim_list = calc_similarity(ftr_mat, same_pairs_idx_list)

    fn_same = osp.join(save_dir, 'same_pairs_similarity.txt')
    fp_same = open(fn_same, 'w')

    for (i, sim) in enumerate(same_sim_list):
        fp_same.write("%s %s %f\n" %
                      (same_img_list[i][0], same_img_list[i][1], sim))
    fp_same.close()

    diff_pairs_idx_list, diff_img_list = load_image_pairs(args.diff_pairs_file)
    diff_sim_list = calc_similarity(ftr_mat, diff_pairs_idx_list)

    fn_diff = osp.join(save_dir, 'diff_pairs_similarity.txt')
    fp_diff = open(fn_diff, 'w')

    for (i, sim) in enumerate(diff_sim_list):
        fp_diff.write("%s %s %f\n" %
                      (diff_img_list[i][0], diff_img_list[i][1], sim))
    fp_diff.close()

    threshs = None
    tp, fn, tn, fp = calc_roc(same_sim_list, diff_sim_list, threshs, save_dir)
    calc_presicion_recall(tp, fn, tn, fp, threshs, save_dir)
    draw_analysis_figure(tp, fn, tn, fp, save_dir, True)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('feature_mat_file', type=str,
                        help='Path to the .mat feature file.')
    parser.add_argument('--same_pairs_file', type=str,
                        help='The file of same pairs list',
                        default='./test_pairs_same.txt')
    parser.add_argument('--diff_pairs_file', type=str,
                        help='The file of diff pairs list',
                        default='./test_pairs_diff.txt')
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
        same_pairs_file = '../lfw_data/test_pairs_same.txt'
        diff_pairs_file = '../lfw_data/test_pairs_diff.txt'

        save_dir = './eval_rlt'
        distance = 'cosine'

        sys.argv.append(ftr_path)
        sys.argv.append('--same_pairs_file={}'.format(same_pairs_file))
        sys.argv.append('--diff_pairs_file={}'.format(diff_pairs_file))
        sys.argv.append('--save_dir={}'.format(save_dir))
        sys.argv.append('--distance=' + distance)

    eval_roc_and_pr(sys.argv[1:])
