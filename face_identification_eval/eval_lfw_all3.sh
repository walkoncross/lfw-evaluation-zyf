save_dir=eval_rlt-lfw
if [ $# -gt 1 ]; then
    save_dir=${save_dir}-$2
fi
nohup python ./eval_top_N.py ${1}.mat --save_dir=./${save_dir}-noflip/ >> nohup-lfw-noflip.out & 
nohup python ./eval_top_N.py ${1}_mirror_eltavg.mat --save_dir=./${save_dir}-eltavg/ >> nohup-lfw-eltavg.out & 
nohup python ./eval_top_N.py ${1}_mirror_eltmax.mat --save_dir=./${save_dir}-eltmax/ >> nohup-lfw-eltmax.out & 
