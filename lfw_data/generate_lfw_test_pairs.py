import os
import os.path as osp
import json

import scipy.io as sio

list_file = './lfw_list_mtcnn.txt'
pairs_mat_file = './lfw_pairs_zyf.mat'

fn_same_pairs = 'test_pairs_same.txt'
fn_diff_pairs = 'test_pairs_diff.txt'


def load_image_list(list_file):
    fp = open(list_file, 'r')

    img_list = []
    id_list = []

    cnt = 0
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

def main():
    img_list, id_list = load_image_list(list_file)

    pairs_mat = sio.loadmat(pairs_mat_file)

    pairs = pairs_mat['pos_pair']
    fp = open(fn_same_pairs, 'w')

    for i in range(3000):
        idx1 = pairs[0, i] - 1
        idx2 = pairs[1, i] - 1
        fp.write("%d %d %d %d %d %s %s\n" %
                           (idx1, idx2, 1,
                            id_list[idx1], id_list[idx2],
                            img_list[idx1], img_list[idx2] )
                           )

    fp.close()

    pairs = pairs_mat['neg_pair']
    fp = open(fn_diff_pairs, 'w')

    for i in range(3000):
        idx1 = pairs[0, i] - 1
        idx2 = pairs[1, i] - 1
        fp.write("%d %d %d %d %d %s %s\n" %
                           (idx1, idx2, 0,
                            id_list[idx1], id_list[idx2],
                            img_list[idx1], img_list[idx2] )
                           )

    fp.close()

if __name__=='__main__':
    main()