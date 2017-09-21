import os
import os.path as osp
import json

import numpy as np
import scipy.io as sio

list_file = './lfw_list_mtcnn.txt'
#pairs_mat_file = './lfw_pairs_zyf.mat'

fn_gt_mat = 'lfw_all_pairs_groundtruth.mat'


def load_image_list(list_file):
    fp = open(list_file, 'r')

    img_list = []
    id_list = []

    for line in fp:
        line=line.strip()
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
        gt_mat[i] = (id_mat==id_mat[i])

    return gt_mat

def main():
    img_list, id_list = load_image_list(list_file)
    num_ids = len(id_list)

    gt_mat = generate_gt_mat(id_list)
    print "gt_mat.shape: ", gt_mat.shape
#    print gt_mat
    gt_mat_sum = gt_mat.sum()
    print "gt_mat.sum: ", gt_mat_sum

    print "pos_pairs ratio: ", (gt_mat_sum - num_ids) / float(num_ids*num_ids - num_ids)

#    mdict = {'all_pairs_gt': gt_mat}
#    sio.savemat(fn_gt_mat, mdict)
#
#    mdict2 = sio.loadmat(fn_gt_mat)
#    gt_mat2 = mdict2['all_pairs_gt']
#
#    print "gt_mat2.shape: ", gt_mat2.shape
#    print "gt_mat2"
#
#    gt_mat2_sum = gt_mat2.sum()
#    print "gt_mat2.sum: ", gt_mat2_sum
#
#    print "pos_pairs ratio: ", (gt_mat2_sum - num_ids) / float(num_ids*num_ids - num_ids)


if __name__=='__main__':
    main()