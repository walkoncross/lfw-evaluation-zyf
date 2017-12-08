save_name=eval_3rlt
if [ $# -gt 0 ]; then
	save_name=${save_name}'-'${1##*/}
fi

save_name=${save_name}.txt
echo 'will save result into '${save_name}

rlt_fn_array=(
nohup-lfw-noflip.txt
nohup-lfw-eltmax.txt
nohup-lfw-eltavg.txt
)

for i in {0..2}; do
	rlt_fn=${rlt_fn_array[$i]}
	
	echo '1.'$i >> $save_name
	echo $cmd >> $save_name
	cat $rlt_fn >> $save_name
	echo '' >> $save_name
done