#!/bin/bash


w_dir=`pwd`

# Nasty way to make temporary storing at scratch
#out=`echo  $w_dir | sed "s:.*/mnt/matylda6/rohdin/expts/runs/x-vec_python_expts::"`
#out="/mnt/scratch06/tmp/rohdin/${out}/"

tmp_dir=$w_dir

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
    else
	echo "WARNING: plda_vad.scp not found. Will not apply VAD."
    fi

    # This is to provide some additional info to the extractor, e.g, domain info
    if [ $side_info != "None" ];then
    	if [ $set_w_side_info == "plda" ];then
	    side_info_string=${side_info_string}:1
    	else
    	    side_info_string=${side_info_string}:0
    	fi	
    fi
   
    output_dir=${tmp_dir}/output
    mkdir -p $output_dir
   
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

    extract_command="python $SCRIPT_DIR/extract_xvectors_gen_kaldi_input.py --out_dir=$output_dir --vad_scp=$plda_vad_scp --model=$model --scp=$job_dir/splits/plda_feats.\${ID} --window_size=1000000 --shift=1000000 --n_cores=1 --extract=$var_to_extract $side_info_string --store_format=h5 --context=22 > $job_dir/logs/extract_xvectors_gen_kaldi_input.py.\${ID}.log 2>&1"
    
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
    else
	echo "WARNING: dev_eval_vad.scp not found. Will not apply VAD."
    fi

    # This is to provide some additional info to the extractor, e.g, domain info
    if [ $side_info != "None" ];then
    	if [ $set_w_side_info == "dev_eval" ];then
	    side_info_string=${side_info_string}:1
    	else
    	    side_info_string=${side_info_string}:0
    	fi	
    fi
   
    output_dir=${tmp_dir}/output
    mkdir -p $output_dir
   
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

    extract_command="python $SCRIPT_DIR/extract_xvectors_gen_kaldi_input.py --out_dir=$output_dir --vad_scp=$dev_eval_vad_scp --model=$model --scp=$job_dir/splits/dev_eval_feats.\${ID} --window_size=1000000 --shift=1000000 --n_cores=1 --extract=$var_to_extract $side_info_string --store_format=h5 --context=22 > $job_dir/logs/extract_xvectors_gen_kaldi_input.py.\${ID}.log 2>&1"
    
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

    exit
    # Collect the embeddings to one h5 file
    echo "python ${SCRIPT_DIR}/collect_embds_h5.py -d ${output_dir} -n plda -s $plda_scp"
    python ${SCRIPT_DIR}/collect_embds_h5.py -d ${output_dir} -n plda -s /mnt/matylda6/rohdin/expts/lists/sre19_lists/plda_train.scp
    
    echo "python ${SCRIPT_DIR}/collect_embds_h5.py -d ${output_dir} -n sre18 -s /mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_dev_cmn2_HZ.enroll.scp,/mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_dev_cmn2_HZ.test.scp,/mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_evl_cmn2_HZ.enroll.scp,/mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_evl_cmn2_HZ.test.scp"
    python ${SCRIPT_DIR}/collect_embds_h5.py -d ${output_dir} -n sre18
    -s /mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_dev_cmn2_HZ.enroll.scp,
    /mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_dev_cmn2_HZ.test.scp,
    /mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_evl_cmn2_HZ.enroll.scp,
    /mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_evl_cmn2_HZ.test.scp,
    /mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_cmn2_evl_tst_HZ.enroll.scp,
    /mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_cmn2_evl_tst_HZ.test.scp,
    /mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_dev_unlabeled_HZ.scp


    # python /mnt/matylda6/rohdin/expts/pytel_py3.7/sid_nn/scripts/collect_embds_h5.py -d /mnt/scratch06/tmp/rohdin/sre19/exp_1/expt_init/output/ -n sre18 -s /mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_dev_cmn2_HZ.enroll.scp,/mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_dev_cmn2_HZ.test.scp,/mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_evl_cmn2_HZ.enroll.scp,/mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_evl_cmn2_HZ.test.scp,/mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_cmn2_evl_tst_HZ.enroll.scp,/mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_cmn2_evl_tst_HZ.test.scp
    # /mnt/matylda6/rohdin/expts/lists/sre19_lists/sre18_dev_unlabeled_HZ.scp


    
    echo "    python $SCRIPT_DIR/evaluate_xvectors_plda_voxceleb.py --work_dir=${out} --lda_dim=150 --eval_conditions=sre18_dev_cmn2_HZ,sre18_evl_cmn2_HZ --key_dir=/mnt/matylda6/rohdin/expts/lists/sre19_lists/ > evaluate_xvectors.log 2>&1 &"
    python $SCRIPT_DIR/evaluate_xvectors_plda_voxceleb.py --work_dir=${out} --lda_dim=150 --eval_conditions=sre18_dev_cmn2_HZ,sre18_evl_cmn2_HZ,sre18_cmn2_evl_tst_HZ --mean_sets=sre18_dev_unlabeled_HZ --key_dir=/mnt/matylda6/rohdin/expts/lists/sre19_lists/ > evaluate_xvectors.log 2>&1 &	
    wait
    echo "cp -r ${out}/results ."
    cp -r ${out}/results .
fi



