"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are pre-calculated and saved into a matlab .mat file.
"""

import numpy as np
import argparse
import sys
import os
import os.path as osp

from load_features import load_mat_features


def load_image_list(list_file):
    print "Load image list form file: ", list_file
    fp = open(list_file, 'r')
    img_list = []
    idx_list = []
    id_list = []

    idx_cnt  = 0
    for line in fp:
        line_split = line.split()
        img_list.append(line_split[0])
        if len(line_split)>2: # for line format: <img_name> <img_idx> <img_id>
            idx_list.append(int(line_split[1]))
            id_list.append(int(line_split[2]))
        else: # for line format: <img_name> <img_id>
            idx_list.append(idx_cnt)
            idx_cnt += 1

            id_list.append(int(line_split[1]))

    fp.close()

    return (img_list, np.array(idx_list), np.array(id_list))


def eval_topN(argv):
    args = parse_arguments(argv)

    print 'args:', args
#    for k,v in args:
#        print "{}: {}".format(k,v)
    top_N = args.top_n

    if top_N<5:
        top_N = 5

    save_dir = args.save_dir
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fn_rlt = osp.join(save_dir, 'ident_test_topN_details.txt')
    fn_rlt2 = osp.join(save_dir, 'ident_test_topN_rlt.txt')

    do_norm = not(args.no_normalize)
    ftr_mat = load_mat_features(args.feature_mat_file, do_norm)

    gallery_img_list, gallery_idx_list,  gallery_id_list= load_image_list(args.gallery_list_file)
    gallery_ftrs = ftr_mat[np.array(gallery_idx_list)]
    print "gallery_ftrs.shape: ", gallery_ftrs.shape

    n_gallery = len(gallery_img_list)
    if top_N>n_gallery:
        top_N = n_gallery

    probe_img_list, probe_idx_list, probe_id_list = load_image_list(args.probe_list_file)
    probe_ftrs = ftr_mat[np.array(probe_idx_list)]
    print "probe_ftrs.shape: ", probe_ftrs.shape

    n_probe = len(probe_img_list)

    similarity_mat = np.dot(probe_ftrs, gallery_ftrs.T)
    print "similarity_mat.shape: ", similarity_mat.shape

    fp_rlt = open(fn_rlt, 'w')

    top_N_cnt_ttl = np.zeros(top_N)

    for i in range(n_probe):
        top_N_cnt = np.zeros(top_N)

        sim_row = similarity_mat[i]
        top_N_idx_idx = np.argsort(-sim_row)[:top_N+1]
        top_one_idx = gallery_idx_list[top_N_idx_idx[0]]

#        if i==0:
#            print "probe img_name: ", probe_img_list[i]
#            print "probe img_idx: ", probe_idx_list[i]
#            print "probe img_id: ", probe_id_list[i]
#
#            print 'sim_row.max: ', sim_row.max()
#
#            print "top_1 img_name: ", gallery_img_list[top_N_idx_idx[0]]
#
#            print "===>Before check top_1's img_idx\n"
#            print "top_N_idx_idx:", top_N_idx_idx
#            print "top_N sim: ", sim_row[top_N_idx_idx]
#            print "top_N_idx:", gallery_idx_list[top_N_idx_idx]
#            print "top_N_id:", gallery_id_list[top_N_idx_idx]

        if top_one_idx == probe_idx_list[i]:
            top_N_idx_idx = top_N_idx_idx[1:top_N+1]
        else:
            top_N_idx_idx = top_N_idx_idx[0:top_N]

#        if i==0:
#            print "===>After check top_1's img_idx\n"
#            print "top_N_idx_idx:", top_N_idx_idx
#            print "top_N sim: ", sim_row[top_N_idx_idx]
#            print "top_N_idx:", gallery_idx_list[top_N_idx_idx]
#            print "top_N_id:", gallery_id_list[top_N_idx_idx]

        for j,idx_idx in enumerate(top_N_idx_idx):
            top_N_cnt[j] += (probe_id_list[i]==gallery_id_list[idx_idx])

#        if i==0:
#            print "top_N_cnt:", top_N_cnt

        for j in range(1, top_N):
            top_N_cnt[j] += top_N_cnt[j-1]
#
#        if i==0:
#            print "top_N_cnt:", top_N_cnt

        fp_rlt.write("%s" % probe_img_list[i])
        for j in range(top_N):
            fp_rlt.write("\t%5d" % top_N_cnt[j])

            top_N_cnt_ttl[j] += (top_N_cnt[j]>0)

        fp_rlt.write("\n")
    fp_rlt.close()

    fp_rlt = open(fn_rlt2, 'w')
    fp_rlt.write("args: \n{}\n\n".format(args))

    fp_rlt.write("Num of gallery features: %d\n" % n_gallery )
    fp_rlt.write("Num of probe features: %d\n\n" % n_probe )

    top_N_ratio = top_N_cnt_ttl / n_probe

    print "top_N_cnt_ttl: ", top_N_cnt_ttl
    print "top_N_ratio: ", top_N_ratio


    fp_rlt.write("TOP_N  \tcnt  \t ratio\n")
    fp_rlt.write("----------------------\n")
    for j in range(top_N):
        fp_rlt.write("top_%d\t%5d\t%5.4f\n" % (j+1, top_N_cnt_ttl[j], top_N_ratio[j]))
    fp_rlt.write("\n")

#    for j in range(top_N):
#        fp_rlt.write("%5d\t" % top_N_cnt_ttl[j])
#    fp_rlt.write("\n")
#
#    for j in range(top_N):
#        fp_rlt.write("%5d\t" % top_N_ratio[j])

    fp_rlt.write("\n")

    fp_rlt.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('feature_mat_file', type=str,
                        help='Path to the .mat feature file.')
    parser.add_argument('--gallery_list_file', type=str,
                        help='The file of gallery list',
                        default='./test_ident_gallery.txt')
    parser.add_argument('--probe_list_file', type=str,
                        help='The file of probe list',
                        default='./test_ident_probe.txt')
    parser.add_argument('--save_dir', type=str,
                        help='directory saving result files',
                        default='./eval_rlt/')
    parser.add_argument('--top_n', type=int,
                        help='eval top_n rank.',
                        default=10)
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

#        gallery_list_file = '../lfw_data/test_ident_gallery.txt'
#        probe_list_file = '../lfw_data/test_ident_probe.txt'
#        save_dir = './eval_rlt'
        gallery_list_file = '../lfw_data/test_ident_gallery_full_list.txt'
        probe_list_file = '../lfw_data/test_ident_probe_full_list.txt'
        save_dir = './eval_rlt_lfw_full_list'

        distance = 'cosine'
        top_n = 10

        sys.argv.append(ftr_path)
        sys.argv.append('--gallery_list_file={}'.format(gallery_list_file))
        sys.argv.append('--probe_list_file={}'.format(probe_list_file))
        sys.argv.append('--save_dir={}'.format(save_dir))
        sys.argv.append('--top_n={}'.format(top_n))
        sys.argv.append('--distance=' + distance)

    eval_topN(sys.argv[1:])
