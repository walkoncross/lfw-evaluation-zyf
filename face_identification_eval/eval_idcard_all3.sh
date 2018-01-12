save_dir=eval_rlt-idcard
if [ $# -gt 1 ]; then
    save_dir=${save_dir}-$2
fi
nohup python ./eval_top_N.py ${1}.mat --gallery_list_file=/disk2/data/FACE/face-idcard-list/idcard_test_ident_gallery.txt --probe_list_file=/disk2/data/FACE/face-idcard-list/idcard_test_ident_probe.txt --save_dir=./${save_dir}-noflip/ >> nohup-idcard-noflip.out & 
nohup python ./eval_top_N.py ${1}_mirror_eltavg.mat --gallery_list_file=/disk2/data/FACE/face-idcard-list/idcard_test_ident_gallery.txt --probe_list_file=/disk2/data/FACE/face-idcard-list/idcard_test_ident_probe.txt --save_dir=./${save_dir}-eltavg/ >> nohup-idcard-eltavg.out & 
nohup python ./eval_top_N.py ${1}_mirror_eltmax.mat --gallery_list_file=/disk2/data/FACE/face-idcard-list/idcard_test_ident_gallery.txt --probe_list_file=/disk2/data/FACE/face-idcard-list/idcard_test_ident_probe.txt --save_dir=./${save_dir}-eltmax/ >> nohup-idcard-eltmax.out & 
