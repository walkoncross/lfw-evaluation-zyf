#!/bin/sh

featfile=$1
savedir=feat

if [ $# -gt 1 ]; then
	save_dir=$2
fi

nohup python ./eval_roc_and_pr.py ${featfile}.mat \
	--same_pairs_file ../lfw-data/test_pairs_same.txt \
	--diff_pairs_file ../lfw-data/test_pairs_diff.txt \
	--save_dir verif_rlt_idcard_bjxj-${save_dir}-noflip \
	--distance cosine > nohup-log-idcard-bjxj-noflip.txt &

nohup python ./eval_roc_and_pr.py ${featfile}_mirror_eltmax.mat \
	--same_pairs_file ../lfw-data/test_pairs_same.txt \
	--diff_pairs_file ../lfw-data/test_pairs_diff.txt \
	--save_dir verif_rlt_idcard_bjxj-${save_dir}-eltmax \
	--distance cosine > nohup-log-idcard-bjxj-eltmax.txt &

nohup python ./eval_roc_and_pr.py ${featfile}_mirror_eltmax.mat \
	--same_pairs_file ../lfw-data/test_pairs_same.txt \
	--diff_pairs_file ../lfw-data/test_pairs_diff.txt \
	--save_dir verif_rlt_idcard_bjxj-${save_dir}-eltavg \
	--distance cosine > nohup-log-idcard-bjxj-eltavg.txt &

