import os
import os.path as osp
import random

import scipy.io as sio

list_file = './img_list_all_with_id.txt'
#pairs_mat_file = './lfw_pairs_zyf.mat'

fn_probe = 'idcard_bjxj_test_ident_probe.txt'
fn_gallery = 'idcard_bjxj_test_ident_gallery.txt'


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

def main():
    img_list, id_list = load_image_list(list_file)

    num_imgs = len(id_list)

    fp_probe = open(fn_probe, 'w')
    fp_gallery = open(fn_gallery, 'w')

    last_id = 0
    cls_cnt = 0
    has_idcard = 0

    for i in range(num_imgs):
        if 'card' in img_list[i].split()[-1]:
            fp_gallery.write('%s %d %d\n' % (img_list[i], i, id_list[i]) )
            has_idcard = 1
        elif id_list[i]!=last_id:
            has_idcard = 0

        if id_list[i]==last_id:
            cls_cnt += 1
        else:
            base_idx = i - cls_cnt

            if cls_cnt>1 and has_idcard:
                for j in range(cls_cnt):
                    if 'card' not in img_list[base_idx+j].split()[-1]:
                        fp_probe.write('%s %d %d\n' % (img_list[base_idx+j], base_idx+j, id_list[base_idx+j]) )

            cls_cnt = 1
            last_id = id_list[i]

    fp_gallery.close()
    fp_probe.close()

if __name__=='__main__':
    main()