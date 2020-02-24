#!/bin/bash

# Extracts the x-vectors with TF, then trains PLDA and evaluates with Kaldi tools.
# List etc. should be in Kaldi format.

SEQUENS_path="some_path/SEQUENS/py3"
kaldi_io="some_path/software/kaldi-io-for-python/"
export PYTHONPATH=${SEQUENS_path}/:${kaldi_io}/
export PATH=${PATH}:some_path/kaldi/src/featbin/:some_path/kaldi/src/ivectorbin/:some_path/kaldi/src/bin/
export train_cmd="some_path//kaldi/egs/wsj/s5/utils/queue.pl --config some_path/conf/queue.johan.conf --mem 4G "

conf_dir="some_path/conf/"

plda_feats_scp="some_other_path/plda_combined_no_sil_no_aug.scp"
plda_vad_scp="None"
dev_eval_feats_scp="some_path/kaldi/egs/voxceleb/v2/data/voxceleb1/feats.scp"
dev_eval_vad_scp="some_path/kaldi/egs/voxceleb/v2/data/voxceleb1/vad.scp"


train_dir="dir_where_training_was_done"
model_dir=$train_dir/output
log_file=$train_dir/train_xvector_extractor_reco_kaldi_feats.log

eval_sets="list_test_all2 list_test_hard2 veri_test2 list_test_all list_test_hard veri_test"
trial_list_dir="some_other_path/tf_xvector_baselines/lists/voxceleb/trials/"

# The variables to extract. If many, separate them with comma.
# But NOTICE, although all of them will be extracted, only the
# first will be used in the evaluation script. Change manually
# there if you want to use another.
var_to_extract="embd_A"
#side_info="S1_p"
#set_w_side_info="sre18"

for l in 0;do
    if [ $l == "init" ];then
	echo "Using initial model"
	model=model-0
	name="init"
    elif [ $l == "last" ];then
	echo "Using last model"
	epoch=`grep Finished $log_file | sed "s/.*Finished epoch: //" | sed "s/\,.*//" | tail -n 1`
	model=model-$epoch
	name="epoch_$epoch"
    else
	echo Lower loss limit: $l
	loss=`grep improved $log_file | sed "s/.*Dev. loss //" | sed "s/ Prev.*//" | awk -v l=${l} '{if ($1 > l) print $1}' | tail -n 1`
	epoch=`grep improved $log_file | grep "Dev. loss $loss "| sed "s/.*Finished epoch: //" | sed "s/,.*//"`
	echo "Lowest loss higher than limit $loss"
	model=model-$epoch
	name="${l}_$loss"
    fi

    echo Model: ${model_dir}${model}
    
    
    if [ ! -d expt_$name ];then
	echo "Running extraction and evaluation"
	mkdir expt_$name
	cd expt_$name

	if [ $plda_feats_scp != "None" ];then  cp $plda_feats_scp plda_feats.scp;fi
	if [ $plda_vad_scp != "None" ];then cp $plda_vad_scp plda_vad.scp;fi
	if [ $dev_eval_feats_scp != "None" ];then cp $dev_eval_feats_scp dev_eval_feats.scp;fi
	if [ $dev_eval_vad_scp != "None" ];then cp $dev_eval_vad_scp dev_eval_vad.scp;fi
	
	rm -r dev_eval_sets.txt dev_eval_cond.txt
	for x in $eval_sets;do
	    echo -n "${trial_list_dir}/${x}.enroll.scp," >> dev_eval_sets.txt
	    echo -n "${trial_list_dir}/${x}.test.scp," >> dev_eval_sets.txt
	    echo $x >> dev_eval_cond.txt
	done
	sed -i 's:,$::' dev_eval_sets.txt  # Remove last comma
	
	cut -f1 -d" "  $plda_feats_scp | sed "s/-/:/" | awk -F":" '{print $1"="$1"-"$2}' > plda.scp
	
	$SEQUENS_path/scripts/extract_collect_and_evaluate_xvectors.sh $model_dir $model $var_to_extract $conf_dir $side_info $set_w_side_info \
								       > extract_collect_and_evaluate_xvectors.sh.log

	$SEQUENS_path/scripts/evalauate_xvec_plda_kaldi.sh output plda some_path/kaldi/egs/voxceleb/v2/data/train_combined_no_sil/utt2spk plda some_path/kaldi/egs/voxceleb/v2/data/train_combined_no_sil/spk2utt None dev_eval_cond.txt some_path/kaldi/egs/voxceleb/v2/voxceleb1_trials 200 > results_plda.txt
	
	cd ..
    fi
    echo
   
done
