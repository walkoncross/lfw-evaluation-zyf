#!/bin/sh

featfile=$1
savedir=./verif_rlt_idcard_bjxj

if [ $# -gt 1 ]; then
	save_dir=$2
fi

nohup python ./eval_roc_and_pr.py ${featfile} \
	--same_pairs_file /disk2/data/FACE/face-idcard-zxt/test_pairs_same_with_id.txt \
	--diff_pairs_file /disk2/data/FACE/face-idcard-zxt/test_pairs_diff_with_id.txt \
	--save_dir ${save_dir} \
	--distance cosine > nohup-log-idcard-bjxj-noflip.txt &

