#!/bin/bash


w_dir=`pwd`

# Nasty way to make temporary storing at scratch
#out=`echo  $w_dir | sed "s:.*/mnt/matylda6/rohdin/expts/runs/x-vec_python_expts::"`
#out="/mnt/scratch06/tmp/rohdin/${out}/"

tmp_dir=$w_dir

output_dir=${tmp_dir}/output
mkdir -p $output_dir

model_dir=$1 
model=$2
var_to_extract=$3
conf_dir=$4
SCRIPT_DIR=`readlink -f $0 | xargs dirname`  # The directory of this script 


if [ $# -ne 4 ] && [ $# -ne 6 ];then
   echo "Usage:    $0 model_dir model_file var_to_extract [side_info] ..."
   echo "Example 1: $0 .../x-vec_python_train/test_61_cleanup_cont/output/ model_feat2score_epoch-9_lr-0.001_lossTr-0.167914892249_lossDev-0.207157.h5 embd_A,embd_B"
   echo "Example 2: $0 .../x-vec_python_train/test_61_cleanup_cont/output/ best (Model with best dev loss is used) embd_A,embd_B"
   exit
fi

if [ $# -eq 6 ] ;then
    side_info=$5
    echo "Using $side_info as side info to TDNN."
    side_info_string="--side_info=$side_info"
    set_w_side_info=$6
    echo "Adding side_info value 1 to $5"
    
else
    echo "No side info is used in TDNN."
    side_info="None"
    side_info_string=""
    side_info_value=""
fi


model=$model_dir/$model
echo "Model: $model"


# Extract x-vectors for plda train data
if [ "A" == "A" ];then

    # Check that feature scp exists
    if [ -e ${w_dir}/plda_feats.scp ];then
	plda_feats_scp=${w_dir}/plda_feats.scp	
    else
	echo "ERROR: plda_feats.scp not found"
	exit -1 
    fi

    # Check whether VAD scp exists. If it doesn't exist, VAD will not be applied.
    if [ -e ${w_dir}/plda_vad.scp ];then
	plda_vad_scp=${w_dir}/plda_vad.scp
	plda_vad_string="--vad_scp=$plda_vad_scp"
    else
	echo "WARNING: plda_vad.scp not found. Will not apply VAD."
	plda_vad_string=""
    fi

    # This is to provide some additional info to the extractor, e.g, domain info
    if [ $side_info != "None" ];then
    	if [ $set_w_side_info == "plda" ];then
	    side_info_string=${side_info_string}:1
    	else
    	    side_info_string=${side_info_string}:0
    	fi	
    fi
      
    scp=$plda_scp

    job_dir=`pwd`/sge_plda
    n_splits_plda=128             # TODO: Set this in a more automatic way?

    rm -r $job_dir
    mkdir $job_dir
    cd $job_dir

    mkdir splits logs    
    cd splits
    split -a 4 -d -n l/$n_splits_plda $plda_feats_scp plda_feats.
    cd ../../

    extract_command="python $SCRIPT_DIR/extract_xvectors_gen_kaldi_input.py --out_dir=$output_dir $plda_vad_string --model=$model --scp=$job_dir/splits/plda_feats.\${ID} --window_size=1000000 --shift=1000000 --n_cores=1 --extract=$var_to_extract $side_info_string --store_format=h5 --context=22 > $job_dir/logs/extract_xvectors_gen_kaldi_input.py.\${ID}.log 2>&1"
    
    cat ${conf_dir}/tf_extract.conf | sed "s:JOB_DIR:${job_dir}:" | sed "s:W_DIR:${w_dir}:" > $job_dir/qsub.sh
    echo  $extract_command >> $job_dir/qsub.sh
fi




# Extract x-vectors for dev and eval data
if [ "A" == "A" ];then

    # Check that feature scp exists
    if [ -e ${w_dir}/dev_eval_feats.scp ];then
	dev_eval_feats_scp=${w_dir}/dev_eval_feats.scp
    else
	echo "ERROR: dev_eval_feats.scp not found"
	exit -1 
    fi

    # Check whether VAD scp exists. If it doesn't exist, VAD will not be applied.
    if [ -e ${w_dir}/dev_eval_vad.scp ];then
	dev_eval_vad_scp=${w_dir}/dev_eval_vad.scp
	dev_eval_vad_string="--vad_scp=$dev_eval_vad_scp"
    else
	echo "WARNING: dev_eval_vad.scp not found. Will not apply VAD."
	dev_eval_vad_string=""
    fi

    # This is to provide some additional info to the extractor, e.g, domain info
    if [ $side_info != "None" ];then
    	if [ $set_w_side_info == "dev_eval" ];then
	    side_info_string=${side_info_string}:1
    	else
    	    side_info_string=${side_info_string}:0
    	fi	
    fi
      
    scp=$dev_eval_scp

    job_dir=`pwd`/sge_dev_eval
    n_splits_dev_eval=128              # TODO: Set this in a more automatic way?

    rm -r $job_dir
    mkdir $job_dir
    cd $job_dir

    mkdir splits logs    
    cd splits
    split -a 4 -d -n l/$n_splits_dev_eval $dev_eval_feats_scp dev_eval_feats.
    cd ../../

    extract_command="python $SCRIPT_DIR/extract_xvectors_gen_kaldi_input.py --out_dir=$output_dir $dev_eval_vad_string --model=$model --scp=$job_dir/splits/dev_eval_feats.\${ID} --window_size=1000000 --shift=1000000 --n_cores=1 --extract=$var_to_extract $side_info_string --store_format=h5 --context=22 > $job_dir/logs/extract_xvectors_gen_kaldi_input.py.\${ID}.log 2>&1"
    
    cat ${conf_dir}/tf_extract.conf | sed "s:JOB_DIR:${job_dir}:" | sed "s:W_DIR:${w_dir}:" > $job_dir/qsub.sh
    echo  $extract_command >> $job_dir/qsub.sh
fi



if [ "A" == "A" ];then
    # "sync" prevents process from exit. "&" allows next jobs to start while first is
    # running. "wait" makes sure both of them has finished before next step starts.
    # (Removing "sync" would make the next step start directly even if wait is there) 
    qsub -t 1-$n_splits_dev_eval -sync yes sge_dev_eval/qsub.sh &
    qsub -t 1-$n_splits_plda -sync yes sge_plda/qsub.sh &
    wait
    echo "Extraction finished at " `date`
    
    mkdir ${tmp_dir}/models ${tmp_dir}/results

    # Check that list of dev_eval sets  exists
    if [ -e ${w_dir}/dev_eval_sets.txt ];then
	dev_eval_sets=$(cat ${w_dir}/dev_eval_sets.txt)
    else
	echo "ERROR: dev_eval_sets.txt not found"
	exit -1 
    fi
    collect_embd_dev_eval_command="python3 ${SCRIPT_DIR}/collect_embds_h5.py -d ${output_dir}/ -n dev_eval_feats -of kaldi -s $dev_eval_sets"
    echo ${collect_embd_dev_eval_command}
    eval ${collect_embd_dev_eval_command} > collect_embd_dev_eval.log 2>&1

    collect_embd_plda_command="python3 ${SCRIPT_DIR}/collect_embds_h5.py -d ${output_dir}/ -n plda_feats -of kaldi -s plda.scp"
    echo ${collect_embd_plda_command}
    eval ${collect_embd_plda_command} > collect_embd_plda.log 2>&1  
    
fi



