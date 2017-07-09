"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import roc


def calc_distance(features1, features2, distance='cosine'):
    print('calc_distance: using {} distance'.format(distance))

    # consinde distance = 1 - (emb1/norm(emb1))*(emb2/norm(emb2)).T
    if distance == 'cosine':
        tmp = features1 * features2
        dist = 1 - np.sum(tmp, 1)
    else:  # default 'squared' distance=diff^2
        diff = np.subtract(features1, features2)
        dist = np.sum(np.square(diff), 1)

#    print 'dist.shape: ', dist.shape

    return dist


def lfw_evaluate(features, actual_issame, distance):
    # calc evaluation metrics
    #    print('Using: {} distance'.format(distance))

    if distance == 'cosine':
        max_thresh = 1.0
        thresh_step = 0.00025
    else:
        max_thresh = 4.0
        thresh_step = 0.001

#    thresholds = np.arange(0, max_thresh, thresh_step)
    thresholds = None

    features1 = features[0]
    features2 = features[1]

#    print('features1.shape: {}'.format(features1.shape))
#    print('features2.shape: {}'.format(features2.shape))

    dist = calc_distance(features1, features2, distance)

    tpr, fpr, best_accuracy = roc.calc_roc(thresholds,
                                           dist,
                                           actual_issame)

#    thresholds = np.arange(0, max_thresh, thresh_step)

    val_far = roc.calc_val(thresholds,
                           dist,
                           actual_issame
                           )
    return tpr, fpr, best_accuracy, val_far


def lfw_evaluate_kfolds(features, actual_issame, distance, nrof_folds=10):
    # calc evaluation metrics
    #    print('Using: {} distance'.format(distance))

    if distance == 'cosine':
        max_thresh = 1.0
        thresh_step = 0.00025
    else:
        max_thresh = 4.0
        thresh_step = 0.001

#    thresholds = np.arange(0, max_thresh, thresh_step)
    thresholds = None

    features1 = features[0]
    features2 = features[1]

#    print('features1.shape: {}'.format(features1.shape))
#    print('features2.shape: {}'.format(features2.shape))
#
    dist = calc_distance(features1, features2, distance)

    tpr, fpr, accuracy = roc.calc_roc_kfolds(thresholds,
                                             dist,
                                             actual_issame)

#    thresholds = np.arange(0, max_thresh, thresh_step)

    val, val_std, far = roc.calc_val_kfolds(thresholds,
                                            dist,
                                            actual_issame,
                                            1e-3)
    return tpr, fpr, accuracy, val, val_std, far


def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(
                lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(
                lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(
                lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(
                lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        # Only add the pair if both paths exist
        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)
