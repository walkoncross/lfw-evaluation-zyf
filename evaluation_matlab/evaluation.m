clc
% load data
% load('../results/LightenedCNN_A_lfw.mat');      % model A
% load('../results/LightenedCNN_B_lfw.mat');      % model B
%load('../results/LightenedCNN_C_lfw.mat');      % model C
load('C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/center_face_model_fixbug.mat');
% load('C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/face_snapshot_0504_val0.15_iter_28000_fixbug.mat');
% load('C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/face_snapshot_0505_val0.1_iter_28000_fixbug.mat');
% load('C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/face_snapshot_0505_val0.1_iter_50000_fixbug.mat');
% load('C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/face_snapshot_0507_val0.1_batch416_iter_50000_fixbug.mat');
% load('C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/face_snapshot_0509_val0.1_batch476_iter_36000_fixbug.mat');

% load('C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/lfw-mtcnn-aligned-224x224_vgg-face_ftr.mat');
% load('C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/lfw-ftr-nowarp-224x224_vgg-face.mat');

% load('C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/LFW-mtcnn-aligned-96x112_center_face_model_orig.mat');
% load('C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/LFW-mtcnn-aligned-96x112_0504_val0.15_iter_28000.mat');


%load('lfw_pairs.mat');
load('lfw_pairs_zyf.mat');

% pos
for i = 1: length(pos_pair)
    feat1 = features(pos_pair(1, i), :)';
    feat2 = features(pos_pair(2, i), :)';
    pos_scores(i) = distance.compute_cosine_score(feat1, feat2);
%     pos_scores(i) = -distance.compute_L2_score(feat1, feat2);
end
pos_label = ones(1, length(pos_pair));

%neg
for i = 1: length(neg_pair)
    feat1 = features(neg_pair(1, i), :)';
    feat2 = features(neg_pair(2, i), :)';
    neg_scores(i) = distance.compute_cosine_score(feat1, feat2);
%     neg_scores(i) = -distance.compute_L2_score(feat1, feat2);
end
neg_label = -ones(1, length(neg_pair));

scores = [pos_scores, neg_scores];
label = [pos_label neg_label];

% ap
ap = evaluation.evaluate('ap', scores, label);

% roc
roc = evaluation.evaluate('roc', scores, label);

% accuracy
acc = evaluation.evaluate('accuracy', scores, label);
 


%% output
fprintf('best accuracy:         %f\n', acc.measure);
fprintf('best threshold (sim): %f\n', acc.extra.bestThresh);
fprintf('best threshold (dist): %f\n', 1.0 - acc.extra.bestThresh);

fprintf('ap:           %f\n', ap.measure);
fprintf('eer:          %f\n', roc.measure);
fprintf('tpr@far=0.01:       %f\n', roc.extra.tpr001*100);
fprintf('tpr@far=0.001:      %f\n', roc.extra.tpr0001*100);
fprintf('tpr@far=0.0001:     %f\n', roc.extra.tpr00001*100);
fprintf('tpr@far=0.00001:    %f\n', roc.extra.tpr000001*100);
fprintf('tpr@far=0:         %f\n', roc.extra.tpr0*100);
result = [ap.measure/100 roc.measure/100  roc.extra.tpr001 roc.extra.tpr0001 roc.extra.tpr00001 roc.extra.tpr000001 roc.extra.tpr0];


