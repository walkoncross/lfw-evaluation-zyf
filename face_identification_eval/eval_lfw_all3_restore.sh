save_dir=eval_rlt-lfw
if [ $# -gt 1 ]; then
    save_dir=${save_dir}-$2
fi
nohup python ./eval_top_N.py ${1}_concat.mat --save_dir=./${save_dir}-concat/ >> nohup-lfw-concat.out & 
nohup python ./eval_top_N.py ${1}_concat_norm.mat --save_dir=./${save_dir}-concat_norm/ >> nohup-lfw-concat_norm.out & 
nohup python ./eval_top_N.py ${1}_avg_norm.mat --save_dir=./${save_dir}-avg_norm/ >> nohup-lfw-avg_norm.out & 
