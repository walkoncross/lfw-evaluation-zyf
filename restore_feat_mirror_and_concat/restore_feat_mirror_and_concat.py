#!/usr/bin/env python

"""
Recover features of flipped images and concat it with features of original images
"""

import numpy as np
import sys
import os
import os.path as osp

from numpy.linalg import norm
import scipy.io as sio

from load_features import load_mat_features

TEST_FEAT_SIM = True


def calc_similarity(feat1, feat2):
    feat1_norm = norm(feat1)
    feat2_norm = norm(feat2)
    print 'feat1_norm: ', feat1_norm
    print 'feat2_norm: ', feat2_norm

    sim = np.dot(feat1, feat2) / (feat1_norm * feat2_norm)

    return sim


def normalize_features(features):
    ftr_norm = norm(features, axis=1)
    ftr_norm = ftr_norm.reshape((-1, 1))
    features = features / ftr_norm

    return features


def main(feat_mat_file, save_dir=None):
    if feat_mat_file.endswith('.mat'):
        base_fn = osp.splitext(feat_mat_file)[0]
    else:
        base_fn = feat_mat_file
        feat_mat_file = base_fn + '.mat'

    if not osp.exists(feat_mat_file):
        print 'File not exists: ', feat_mat_file

    avg_mat_fn = base_fn + '_mirror_eltavg.mat'
    if not osp.exists(avg_mat_fn):
        print 'File not exists: ', avg_mat_fn

    if save_dir and not osp.isdir(save_dir):
        os.makedirs(save_dir)

    orig_ftr_mat = load_mat_features(feat_mat_file, False)
    avg_ftr_mat = load_mat_features(avg_mat_fn, False)
    print 'orignal feat mat shape: ', orig_ftr_mat.shape
    print 'average feat mat shape: ', avg_ftr_mat.shape

    flip_mat_fn = base_fn + '_flip.mat'
    if save_dir:
        flip_mat_fn = osp.join(save_dir, osp.basename(flip_mat_fn))

    flip_ftr_mat = avg_ftr_mat * 2.0 - orig_ftr_mat
    print 'flipped feat mat shape: ', flip_ftr_mat.shape
    sio.savemat(flip_mat_fn, {'features': flip_ftr_mat})

    if TEST_FEAT_SIM:
        for i in range(10):
            sim = calc_similarity(orig_ftr_mat[i], flip_ftr_mat[i])
            print '===> %d pair of (orignal, flipped) featur: ' % i
            print 'sim = ', sim

    # concat original feature and flip feature
    concat_mat_fn = base_fn + '_concat.mat'
    if save_dir:
        concat_mat_fn = osp.join(save_dir, osp.basename(concat_mat_fn))

    concat_ftr_mat = np.hstack((orig_ftr_mat, flip_ftr_mat))
    print 'concat feat mat shape: ', concat_ftr_mat.shape
    sio.savemat(concat_mat_fn, {'features': concat_ftr_mat})

    # normalize features
    orig_ftr_mat = normalize_features(orig_ftr_mat)
    flip_ftr_mat = normalize_features(flip_ftr_mat)

    # concat normalized original feature and flip feature
    concat_norm_mat_fn = base_fn + '_concat_norm.mat'
    if save_dir:
        concat_norm_mat_fn = osp.join(save_dir, osp.basename(concat_norm_mat_fn))

    concat_norm_ftr_mat = np.hstack((orig_ftr_mat, flip_ftr_mat))
    print 'concat norm feat mat shape: ', concat_norm_ftr_mat.shape
    sio.savemat(concat_norm_mat_fn, {'features': concat_norm_ftr_mat})

    # calc average of normalized original feature and flip feature
    avg_norm_mat_fn = base_fn + '_avg_norm.mat'
    if save_dir:
        avg_norm_mat_fn = osp.join(save_dir, osp.basename(avg_norm_mat_fn))

    avg_norm_ftr_mat = (orig_ftr_mat + flip_ftr_mat) * 0.5
    print 'avg norm feat mat shape: ', avg_norm_ftr_mat.shape
    sio.savemat(avg_norm_mat_fn, {'features': avg_norm_ftr_mat})


if __name__ == '__main__':
    save_dir = './restored_flip_and_concat_feats'
    if len(sys.argv) < 2:
        ftr_path = '/disk2/zhaoyafei/lfw-evaluation-zyf/extract_face_features/lfw-simaligned-sphere20-webasian-bs512-56k-1030.mat'
    else:
        ftr_path = sys.argv[1]

    if len(sys.argv) > 2:
        save_dir = sys.argv[2]

    main(ftr_path, save_dir)
