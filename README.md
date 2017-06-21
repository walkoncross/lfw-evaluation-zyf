# LFW evaluation

## 1. extract face features
python requirements: [requirements.txt](extract_face_features/requirements.txt)  
usage:
```
python caffe_ftr.py network_def trained_model image_dir image_list_file layer_name save_file
```
example:
```
nohup python caffe_ftr.py /opt/caffe/face_example/face_deploy.prototxt /opt/caffe/face_example/face_snapshot_0509_val0.1_batch476/face_train_test_iter_36000.caffemodel /disk2/data/FACE/LFW/lfw-aligned-mtcnn/ /disk2/data/FACE/LFW/lfw-aligned-mtcnn/list.txt fc5 face_snapshot_0509_val0.1_batch476_iter_36000.mat &
```

## 2. LFW 10-folds evaluation
python requirements: [requirements.txt](evaluation_10folds/requirements.txt)  

```
cd evaluation_10folds
python validate_on_lfw.py  face_snapshot_0509_val0.1_batch476_iter_36000.mat
```

## 3. (_optional_) accuracy evaluation (Matlab code)
(code from https://github.com/AlfredXiangWu/face_verification_experiment)

In Matlab, run [evaluation_matlab/evaluation.m](evaluation_matlab/evaluation.m).