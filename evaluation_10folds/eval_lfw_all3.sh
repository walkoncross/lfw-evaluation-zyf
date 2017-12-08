save_name=eval_rlt
if [ $# -gt 1 ]; then
    save_name=${save_name}-$2
else
	save_name=${save_name}'-'${1##*/}
fi

save_name=${save_name}.txt
echo 'will save result into '${save_name}
cmd_array=(
	"python ./validate_on_lfw.py ${1}.mat"
	"python ./validate_on_lfw.py ${1}_mirror_eltmax.mat"
	"python ./validate_on_lfw.py ${1}_mirror_eltavg.mat"
)
rlt_fn_array=(
nohup-lfw-noflip.txt
nohup-lfw-eltmax.txt
nohup-lfw-eltavg.txt
)

for i in {0..2}; do
	cmd=${cmd_array[$i]}
	rlt_fn=${rlt_fn_array[$i]}
	nohup $cmd > $rlt_fn &
#	echo 'sleep 5s, wait for eval to finish'
#	sleep 5
done

function combine_3rlt()
{
	#echo 'sleep 15s, wait for 3 eval to finish'
	sleep 15

	#echo 'combine 3 results'

	if [ -f $save_name ]; then
		mv $save_name $save_name.bk
	fi

	for i in {0..2}; do
		cmd=${cmd_array[$i]}
		rlt_fn=${rlt_fn_array[$i]}

		echo '1.'$i >> $save_name
		echo $cmd >> $save_name
		cat $rlt_fn >> $save_name
		echo ' ' >> $save_name
	done
}

#run in background
combine_3rlt & 

#cmd1="python ./validate_on_lfw.py ${1}.mat"
#cmd2="python ./validate_on_lfw.py ${1}_mirror_eltmax.mat"
#cmd3="python ./validate_on_lfw.py ${1}_mirror_eltavg.mat"
#
#rlt1=nohup-lfw-noflip.txt
#rlt2=nohup-lfw-eltmax.txt
#rlt3=nohup-lfw-eltavg.txt
#nohup $cmd1 > $rlt1 &
#nohup $cmd2 > $rlt2 &
#nohup $cmd3 > $rlt3 &
#
#echo 'sleep, wait for last 3 'nohup' to finish'
#sleep 10
#
#echo 'combine 3 results'
#
#echo '1.0' > $save_name
#echo $cmd1 >> $save_name
#cat nohup-lfw-noflip.txt >> $save_name
#
#echo '1.1' >> $save_name
#echo $cmd2 >> $save_name
#cat nohup-lfw-eltmax.txt >> $save_name
#
#echo '1.2' >> $save_name
#echo $cmd3 >> $save_name
#cat nohup-lfw-eltavg.txt >> $save_name#