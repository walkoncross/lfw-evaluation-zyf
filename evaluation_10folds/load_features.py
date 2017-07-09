#! /usr/bin/env python

import numpy as np
from scipy.io import loadmat
from scipy.linalg import norm


def load_mat_features(data_mat_file, pairs_mat_file,
                      nrof_kfolds=10,
                      nrof_features_per_fold=600):
    nrof_features_per_fold /= 2
#    print nrof_features_per_fold

    pairs = loadmat(pairs_mat_file)
    data = loadmat(data_mat_file)

    pos_pairs = pairs['pos_pair'] - 1
    neg_pairs = pairs['neg_pair'] - 1
#    print "pos_pairs.shape: ", pos_pairs.shape
#    print "neg_pairs.shape: ", neg_pairs.shape

    assert(pos_pairs.shape == neg_pairs.shape)
    assert(pos_pairs.shape[0] == 2)
    assert(neg_pairs.shape[0] == 2)
    assert(pos_pairs.shape[1] == nrof_features_per_fold * nrof_kfolds)
    assert(neg_pairs.shape[1] == nrof_features_per_fold * nrof_kfolds)

    features = data['features']
#    print "features.shape: ", features.shape

    pos_ind_1 = pos_pairs[0].reshape(nrof_kfolds, nrof_features_per_fold)
    pos_ind_2 = pos_pairs[1].reshape(nrof_kfolds, nrof_features_per_fold)

#    print pos_ind_1.shape
#    print pos_ind_2.shape

    neg_ind_1 = neg_pairs[0].reshape(nrof_kfolds, nrof_features_per_fold)
    neg_ind_2 = neg_pairs[1].reshape(nrof_kfolds, nrof_features_per_fold)

#    print neg_ind_1.shape
#    print neg_ind_2.shape

    ind_1 = np.hstack((pos_ind_1, neg_ind_1)).reshape(
        nrof_features_per_fold * nrof_kfolds * 2)
    ind_2 = np.hstack((pos_ind_2, neg_ind_2)).reshape(
        nrof_features_per_fold * nrof_kfolds * 2)

#    print ind_1.shape
#    print ind_2.shape

    ftrs1 = features[ind_1]
    ftrs2 = features[ind_2]

#    print ftrs1.shape
#    print ftrs2.shape

    ftr_norm = norm(ftrs1, axis=1)
    ftr_norm = ftr_norm.reshape((-1,1))
    ftrs1 = ftrs1 / ftr_norm

    ftr_norm = norm(ftrs2, axis=1)
    ftr_norm = ftr_norm.reshape((-1,1))
    ftrs2 = ftrs2 / ftr_norm

#    print ftrs1.shape
#    print ftrs1.shape
#    print norm(ftrs1, axis=0)
#    print norm(ftrs2, axis=0)

    ftrs = np.concatenate(
        (ftrs1[np.newaxis, ...], ftrs2[np.newaxis, ...])
    )

#     tmp = np.hstack((np.ones(nrof_features_per_fold),
#                      np.zeros(nrof_features_per_fold)))
# #    tmp = tmp.reshape((nrof_features_per_fold * 2, 1))
#     gt_labels = np.tile(tmp, nrof_kfolds)
    gt_labels = ([1] * nrof_features_per_fold + [0] *
                 nrof_features_per_fold) * nrof_kfolds
    gt_labels = np.array(gt_labels)

    return (ftrs, gt_labels)


if __name__ == "__main__":
    data_mat_file = r'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/face_snapshot_0509_val0.1_batch476_iter_36000_fixbug.mat'
    pairs_mat_file = './lfw_pairs_zyf.mat'
    (ftrs, gt_labels) = load_mat_features(data_mat_file,
                                          pairs_mat_file,
                                          nrof_kfolds=10,
                                          nrof_features_per_fold=600)

    print('ftrs.shape: {}'.format(ftrs.shape))
    print('gt_labels.shape: {}'.format(gt_labels.shape))
#    print('len(gt_labels): {}'.format(len(gt_labels)))
