"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are pre-calculated and saved into a matlab .mat file.
"""

import numpy as np
import argparse
import sys

from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

from load_features import load_mat_features
from lfw_evaluate import lfw_evaluate, lfw_evaluate_kfolds

import matplotlib.pyplot as plt


def get_auc(fpr, tpr):
    auc = metrics.auc(fpr, tpr)

    return auc


def get_eer(fpr, tpr):
    x_min = fpr.min()
    x_max = fpr.max()

    f = interpolate.interp1d(fpr, tpr)
    eer = brentq(lambda x: 1. - x - f(x), x_min, x_max)

    return eer


def main(args):
    print 'args:', args
#    for k,v in args:
#        print "{}: {}".format(k,v)

    ftr_mat, actual_issame = load_mat_features(args.feature_mat_file,
                                                 args.lfw_pairs_mat_file,
                                                 args.lfw_nrof_folds,
                                                 args.nrof_features_per_fold)

    print('ftr_mat.shape: {}'.format(ftr_mat.shape))
    print('actual_issame.shape: {}'.format(actual_issame.shape))

    print('\n=======================================')
    print('EVALUATION WITHOUT K-FOLDS')

    tpr, fpr, best_accuracy, val_far = lfw_evaluate(ftr_mat,
                                                    actual_issame,
                                                    distance=args.distance)

    if args.draw_roc:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title('ROC without K-folds')
        plt.show()

    print('Accuracy: %2.5f @distance_threshold=%2.5f, VAL=%2.5f, FAR=%2.5f'
          % (best_accuracy['accuracy'],
             best_accuracy['threshold'],
             best_accuracy['VAL'],
             best_accuracy['FAR']
             )
          )

    auc = get_auc(fpr, tpr)
    print('Area Under Curve (AUC): %2.5f' % auc)

    eer = get_eer(fpr, tpr)
    print('Equal Error Rate (EER): %2.5f with accuracy=%2.5f' % (eer, 1.0-eer))

    for it in val_far:
        print('Validation rate: %2.5f @ FAR=%2.5f with theshold %2.5f' %
              (it['VAL'], it['FAR'], it['threshold']))

    print('\n=======================================')
    print('EVALUATION WITH K-FOLDS')
    tpr, fpr, accuracy, val, val_std, far = lfw_evaluate_kfolds(ftr_mat,
                                                                actual_issame,
                                                                distance=args.distance,
                                                                nrof_folds=args.lfw_nrof_folds)
    if args.draw_roc:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title('ROC with K-folds')
        plt.show()

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = get_auc(fpr, tpr)
    print('Area Under Curve (AUC): %2.5f' % auc)

    eer = get_eer(fpr, tpr)
    print('Equal Error Rate (EER): %2.5f with accuracy=%2.5f' % (eer, 1.0-eer))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('feature_mat_file', type=str,
                        help='Path to the .mat feature file.')
    parser.add_argument('--distance', type=str,
                        help='how to calc distance: <cosine or squared>.',
                        default='cosine')
    parser.add_argument('--lfw_pairs_mat_file', type=str,
                        help='The file containing the pairs to use for validation.',
                        default='./lfw_pairs_zyf.mat')
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.',
                        default=10)
    parser.add_argument('--nrof_features_per_fold', type=int,
                        help='Number of pairs in each fold.',
                        default=600)
    parser.add_argument('--draw_roc', type=int,
                        help='draw ROC using matplot or not.',
                        default=0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        ftr_path = r'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/center_face_model_fixbug.mat'
#        ftr_path = 'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/face_snapshot_0504_val0.15_iter_28000_fixbug.mat'
#        ftr_path = 'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/face_snapshot_0505_val0.1_iter_28000_fixbug.mat'
#        ftr_path = 'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/face_snapshot_0505_val0.1_iter_50000_fixbug.mat'
#        ftr_path = 'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/face_snapshot_0507_val0.1_batch416_iter_50000_fixbug.mat'
#        ftr_path = 'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/face_snapshot_0509_val0.1_batch476_iter_36000_fixbug.mat'

#        ftr_path = 'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/lfw-mtcnn-aligned-224x224_vgg-face_ftr.mat'
#        ftr_path = 'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/lfw-ftr-nowarp-224x224_vgg-face.mat'
#
#        ftr_path = 'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/LFW-mtcnn-aligned-96x112_center_face_model_orig.mat'
#        ftr_path = 'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/LFW-mtcnn-aligned-96x112_0504_val0.15_iter_28000.mat'

        distance = 'cosine'
        draw_roc = 0

        sys.argv.append(ftr_path)
        sys.argv.append('--distance=' + distance)
        sys.argv.append('--draw_roc={}'.format(draw_roc))

    main(parse_arguments(sys.argv[1:]))
