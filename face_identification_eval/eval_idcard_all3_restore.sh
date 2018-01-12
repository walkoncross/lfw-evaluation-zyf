save_dir=eval_rlt-idcard
if [ $# -gt 1 ]; then
    save_dir=${save_dir}-$2
fi
nohup python ./eval_top_N.py ${1}_concat.mat --gallery_list_file=/disk2/zhaoyafei/facex-sim-test/idcard_test_ident_gallery.txt --probe_list_file=/disk2/zhaoyafei/facex-sim-test/idcard_test_ident_probe.txt --save_dir=./${save_dir}-concat/ >> nohup-idcard-concat.out & 
nohup python ./eval_top_N.py ${1}_concat_norm.mat --gallery_list_file=/disk2/zhaoyafei/facex-sim-test/idcard_test_ident_gallery.txt --probe_list_file=/disk2/zhaoyafei/facex-sim-test/idcard_test_ident_probe.txt --save_dir=./${save_dir}-concat_norm/ >> nohup-idcard-concat_norm.out & 
nohup python ./eval_top_N.py ${1}_avg_norm.mat --gallery_list_file=/disk2/zhaoyafei/facex-sim-test/idcard_test_ident_gallery.txt --probe_list_file=/disk2/zhaoyafei/facex-sim-test/idcard_test_ident_probe.txt --save_dir=./${save_dir}-avg_norm/ >> nohup-idcard-avg_norm.out & 
