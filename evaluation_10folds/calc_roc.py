#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])

#    thresholds = np.zeros(actual_issame.shape)

    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

#    thresholds = np.sort(dist)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        print("===>Kfold: fold_idx=%d" % fold_idx)
#        thresholds = np.sort(dist[train_set])
#        nrof_thresholds = len(thresholds)
#        print("nrof_thresholds=%d" % nrof_thresholds)
#
#        print 'train_set: ', train_set
#        print 'dist.shape: ', dist.shape
#        print 'actual_issame.shape: ', actual_issame.shape
#

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)

        print("\t best_threshold_index = %d" % (best_threshold_index))
        print("\t best_threshold = %f" % (thresholds[best_threshold_index]))

#        thresholds = np.sort(dist[test_set])
#        nrof_thresholds = len(thresholds)
#        print("nrof_thresholds=%d" % nrof_thresholds)

#        print 'test_set: ', test_set
#        print 'dist.shape: ', dist.shape
#        print 'actual_issame.shape: ', actual_issame.shape

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])

        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)

#        print("\t tpr = {}".format(tpr))
#        print("\t fpr = {}".format(fpr))
#        print("\t accuracy = {}".format(accuracy))

    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)

#    print("\n---> In calculate_accuracy:")
#    print("\t threshold = %f" % (threshold))
#    print 'predict_issame.shape: \n', predict_issame.shape
#    print 'actual_issame.shape: \n', actual_issame.shape

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    tn = np.sum(np.logical_and(np.logical_not(
        predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

#    print("\t tp = %d" % (tp))
#    print("\t fp = %d" % (fp))
#    print("\t tn = %d" % (tn))
#    print("\t fn = %d" % (fn))
#    print("\t dist.size = %d" % (dist.size))
#
#    print("\t tpr = %f" % (tpr))
#    print("\t fpr = %f" % (fpr))
#    print("\t accuracy = %f" % (acc))

    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
#        thresholds = np.sort(dist[train_set])
#        nrof_thresholds = len(thresholds)
#        print("nrof_thresholds=%d" % nrof_thresholds)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(
        predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far
