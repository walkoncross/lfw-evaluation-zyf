#!/bin/sh

featfile=$1
savedir=feat

if [ $# -gt 1 ]; then
	save_dir=$2
fi

nohup python ./eval_roc_and_pr.py ${featfile}.mat \
	--save_dir verif_rlt_idcard-${save_dir}-noflip \
	--distance cosine > nohup-log-idcard-noflip.txt &

nohup python ./eval_roc_and_pr.py ${featfile}_mirror_eltmax.mat \
	--save_dir verif_rlt_idcard-${save_dir}-eltmax \
	--distance cosine > nohup-log-idcard-eltmax.txt &

nohup python ./eval_roc_and_pr.py ${featfile}_mirror_eltmax.mat \
	--save_dir verif_rlt_idcard-${save_dir}-eltavg \
	--distance cosine > nohup-log-idcard-eltavg.txt &

