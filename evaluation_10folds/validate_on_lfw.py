"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are pre-calculated and saved into a matlab .mat file.
"""

import numpy as np
import argparse
import sys

from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

import lfw
from load_features import load_mat_features


def main(args):
    print 'args:', args
#    for k,v in args:
#        print "{}: {}".format(k,v)

    emb_array, actual_issame = load_mat_features(args.feature_mat_file,
                                                 args.lfw_pairs_mat_file,
                                                 args.lfw_nrof_folds,
                                                 args.nrof_features_per_fold)

    print('emb_array.shape: {}'.format(emb_array.shape))
    print('actual_issame.shape: {}'.format(actual_issame.shape))
#    print('len(actual_issame): {}'.format(len(actual_issame)))

    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
        actual_issame, nrof_folds=args.lfw_nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('feature_mat_file', type=str,
        help='Path to the .mat feature file.')
    parser.add_argument('--lfw_pairs_mat_file', type=str,
        help='The file containing the pairs to use for validation.', default='./lfw_pairs_zyf.mat')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--nrof_features_per_fold', type=int,
        help='Number of pairs in each fold.', default=600)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
