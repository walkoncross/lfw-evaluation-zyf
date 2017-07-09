#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate


def sorted_nparray_last_argmin(arr, min_val=None):
    if min_val is None:
        min_val = arr.min()

    for idx in range(len(arr)):
        if arr[idx] > min_val:
            min_idx = idx - 1
            break

    return min_idx


def sorted_nparray_first_argmax(arr, max_val=None):
    if max_val is None:
        max_val = arr.max()

    for idx in np.arange(len(arr) - 1, -1, -1):
        if arr[idx] < max_val:
            max_idx = idx + 1
            break

    return max_idx


def calc_accuracy(threshold, dist, actual_issame):
    """
        Calculate accuracy at some threshold
    """

    predict_issame = np.less(dist, threshold)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    tn = np.sum(np.logical_and(np.logical_not(
        predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

    return tpr, fpr, acc


def calc_val_far(threshold, dist, actual_issame):
    """
        Calculate VAL(=TPR) at some threshold
    """

    predict_issame = np.less(dist, threshold)

    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))

    false_accept = np.sum(np.logical_and(
        predict_issame, np.logical_not(actual_issame)))

    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))

    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)

    return val, far


def calc_roc(thresholds,
             dist,
             actual_issame):
    """
        ROC evaluation without k-folds
    """

    if thresholds is None or thresholds == []:
        thresholds = np.sort(dist)

    nrof_thresholds = len(thresholds)

    tprs = np.zeros(nrof_thresholds)
    fprs = np.zeros(nrof_thresholds)

    accuracies = np.zeros(nrof_thresholds)

    # Find the best threshold
    for idx, threshold in enumerate(thresholds):
        tprs[idx], fprs[idx], accuracies[idx] = calc_accuracy(
            threshold, dist, actual_issame)

    best_thresh_idx = np.argmax(accuracies)
    best_thresh = thresholds[best_thresh_idx]
    accuracy = accuracies[best_thresh_idx]

#    print("\t best_thresh_idx = %d" % (best_thresh_idx))
#    print("\t best_threshold = %2.5f" % (thresholds[best_thresh_idx]))
#    print("\t accuracy = %2.5f" % (accuracy))
#

    val, far = calc_val_far(best_thresh, dist, actual_issame)

    best_accuracy = {
        "accuracy": accuracy,
        "threshold": best_thresh,
        "VAL": val,
        "FAR": far
    }

    return tprs, fprs, best_accuracy


def calc_val(thresholds,
             dist,
             actual_issame,
             far_targets=None):
    """
        Calculate VAL(=TPR) at target FARs without k-folds
    """

    if thresholds is None or thresholds == []:
        thresholds = np.sort(dist)

    if not far_targets:
        far_targets = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0]

    nrof_thresholds = len(thresholds)

    val_array = np.zeros(nrof_thresholds)
    far_array = np.zeros(nrof_thresholds)

    # Find the threshold that gives FAR = far_target
    for idx, threshold in enumerate(thresholds):
        val_array[idx], far_array[idx] = calc_val_far(
            threshold, dist, actual_issame)

#    min_far_idx = np.argmin(far_array)
#    max_far_idx = np.argmax(far_array)
#    min_far = far_array[min_far_idx]
#    max_far = far_array[max_far_idx]
#    min_far = np.min(far_array)
#    max_far = np.max(far_array)

    # find the last idx with min_far
#    for idx in range(len(far_array)):
#        if far_array[idx] > min_far:
#            min_far_idx = idx - 1
#            break
    min_far_idx = sorted_nparray_last_argmin(far_array)

    # find the first idx with max_far
#    for idx in np.arange(len(far_array) - 1, -1, -1):
#        if far_array[idx] < max_far:
#            max_far_idx = idx + 1
#            break
    max_far_idx = sorted_nparray_first_argmax(far_array)
    min_far = far_array[min_far_idx]
    max_far = far_array[max_far_idx]

#    print('min_far=%2.5f, max_far=%2.5f' % (min_far, max_far))
#    print('min_far_idx=%d, max_far_idx=%d' % (min_far_idx, max_far_idx))

    f1 = interpolate.interp1d(far_array[min_far_idx:max_far_idx],
                              val_array[min_far_idx:max_far_idx])
    f2 = interpolate.interp1d(far_array[min_far_idx:max_far_idx],
                              thresholds[min_far_idx:max_far_idx],
                              kind='slinear')

    outputs = []

    for idx, far_t in enumerate(far_targets):
        #        print('Calc VAL and threshold @ FAR=%2.5f' % far_t)
        if far_t >= max_far:
            val = val_array[max_far_idx]
            thresh = thresholds[max_far_idx]
        elif far_t <= min_far:
            val = val_array[min_far_idx]
            thresh = thresholds[min_far_idx]
        else:
            val = f1(far_t)
            thresh = f2(far_t)

        t_dict = {
            'FAR': far_t,
            'VAL': val.tolist(),
            'threshold': thresh.tolist()
        }

#        print t_dict

        outputs.append(t_dict)

    return outputs


def calc_roc_kfolds(thresholds,
                    dist,
                    actual_issame,
                    distance='cosine',
                    nrof_folds=10):
    """
        ROC evaluation with k-folds
    """

    if thresholds is None or thresholds == []:
        thresholds = np.sort(dist)

    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))

    nrof_pairs = min(len(actual_issame), dist.shape[0])

    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #        print("===>Kfold: fold_idx=%d" % fold_idx)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for idx, threshold in enumerate(thresholds):
            _, _, acc_train[idx] = calc_accuracy(
                threshold, dist[train_set], actual_issame[train_set])

        best_thresh_idx = np.argmax(acc_train)

#        print("\t best_thresh_idx = %d" % (best_thresh_idx))
#        print("\t best_threshold = %2.5f" % (thresholds[best_thresh_idx]))

        for idx, threshold in enumerate(thresholds):
            tprs[fold_idx, idx], fprs[fold_idx, idx], _ = calc_accuracy(
                threshold, dist[test_set], actual_issame[test_set])

        _, _, accuracy[fold_idx] = calc_accuracy(
            thresholds[best_thresh_idx], dist[test_set], actual_issame[test_set])

#        print("\t accuracy = %2.5f" % (accuracy[fold_idx]))

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)

    return tpr, fpr, accuracy


def calc_val_kfolds(thresholds,
                    dist,
                    actual_issame,
                    far_target,
                    distance='cosine',
                    nrof_folds=10):
    """
        Calculate VAL(=TPR) at target FARs with k-folds
    """

    if thresholds is None or thresholds == []:
        thresholds = np.sort(dist)

    nrof_thresholds = len(thresholds)

    nrof_pairs = min(len(actual_issame), dist.shape[0])
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        far_train = np.zeros(nrof_thresholds)

        for idx, threshold in enumerate(thresholds):
            _, far_train[idx] = calc_val_far(
                threshold, dist[train_set], actual_issame[train_set])

        min_far_idx = sorted_nparray_last_argmin(far_train)
        max_far_idx = sorted_nparray_first_argmax(far_train)
        min_far = far_train[min_far_idx]
        max_far = far_train[max_far_idx]

        if far_target >= max_far:
            threshold = 0.0
        elif far_target <= min_far:
            threshold = thresholds[min_far_idx]
        else:
            f = interpolate.interp1d(far_train[min_far_idx:max_far_idx],
                                     thresholds[min_far_idx:max_far_idx],
                                     kind='slinear')
            threshold = f(far_target)

        val[fold_idx], far[fold_idx] = calc_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)

    return val_mean, val_std, far_mean
