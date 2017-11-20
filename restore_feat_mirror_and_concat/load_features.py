#! /usr/bin/env python

import numpy as np
from scipy.io import loadmat
from scipy.linalg import norm


def load_mat_features(data_mat_file, do_norm=True):
    data = loadmat(data_mat_file)

    if 'features' in data:
        features = data['features']
    elif 'feature' in data:
        features = data['feature']
        features = features.T
    else:
        raise Exception(
            'Counld not find keyword "feature" or "features" in .mat file')

#    print "features.shape: ", features.shape

    if do_norm:
        #        print "Normalize features"
        ftr_norm = norm(features, axis=1)
        ftr_norm = ftr_norm.reshape((-1, 1))
        features = features / ftr_norm

    return features


if __name__ == "__main__":
    data_mat_file = r'C:/zyf/dnn_models/face_models/lfw_eval_results/center_face_model_fixbug.mat'

    ftrs = load_mat_features(data_mat_file)

    print('ftrs.shape: {}'.format(ftrs.shape))
