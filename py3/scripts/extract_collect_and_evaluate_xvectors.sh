#!/bin/bash


w_dir=`pwd`

# Nasty way to make temporary storing at scratch
#out=`echo  $w_dir | sed "s:.*/mnt/matylda6/rohdin/expts/runs/x-vec_python_expts::"`
#out="/mnt/scratch06/tmp/rohdin/${out}/"

tmp_dir=$w_dir

output_dir=${tmp_dir}/output
mkdir -p $output_dir

if [ $# -le 4 ];then
    echo "ERROR $0: Number of input argument should be at least 4 "
    echo "USAGE: $0 model var_to_extract conf_dir [options]"
    echo "EXAMPLE: $0 /workspace/jrohdin/expts/resnet_baselines/exp_2/outputmodel-113 embd_ /workspace/jrohdin/conf/ --side_info S1_p set_w_side_info sre18"
    exit -1
fi

model=$1
var_to_extract=$2
conf_dir=$3
SCRIPT_DIR=`readlink -f $0 | xargs dirname`  # The directory of this script 


options=$(getopt  -o d  --long side_info: --long set_w_side_info: --long arch: --long n_jobs_plda: --long n_jobs_dev_eval: --long use_gpu: --long instance_norm: -- "$@")
eval set -- $options
echo $options

# Default arguments
side_info="None"
side_info_string=""
set_w_side_info="all"
arch="tdnn"
n_jobs_plda=128
n_jobs_dev_eval=128
use_gpu=false
option_string=""
instance_norm=false
while true; do
    case $1 in
    --side_info)
		shift 1
		side_info=$1
		echo "Using $side_info as side info to TDNN."
		side_info_string="--side_info=$side_info"
		;;	
    --set_w_side_info)
		shift 1
		set_w_side_info=$1
		;;		
    --arch)
		shift 1
		arch=$1
		option_string="$option_string --architecture=$arch"	
		;;
    --n_jobs_plda)
		shift 1
		n_jobs_plda=$1
		;;
    --n_jobs_dev_eval)
		shift 1
		n_jobs_dev_eval=$1
		;;
    --use_gpu)
		shift 1
		use_gpu=$1
		if [ $use_gpu = "true" ];then
		    option_string="$option_string --use_gpu"
		fi
		;;
    --instance_norm)
		shift 1
		instance_norm=$1
		if [ $instance_norm = "true" ];then
		    option_string="$option_string --instance_norm"
		fi
		;;
    --)
	shift
	break
	;;
    esac
    shift
done


echo model:               $model
echo var_to_extract:      $var_to_extract
echo conf_dir:            $conf_dir
echo side_info:           $side_info
echo set_w_side_info:     $set_w_side_info
echo arch:                $arch
echo n_jobs_plda:         $n_jobs_plda
echo n_jobs_dev_eval:     $n_jobs_dev_eval
echo use_gpu:             $use_gpu
echo instance_norm: $instance_norm

echo option_string:   $option_string

if [ $n_jobs_plda -le 0 ];then
    echo "Skipping PLDA"
else
    echo "Extract x-vectors for plda train data"
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
    	if [[ $set_w_side_info == "plda" ]] || [[ $set_w_side_info == "all" ]];then
	    side_info_string=${side_info_string}:1
    	else
    	    side_info_string=${side_info_string}:0
    	fi	
    fi
      
    scp=$plda_scp

    job_dir=`pwd`/sge_plda


    rm -r $job_dir
    mkdir $job_dir
    cd $job_dir

    mkdir splits logs    
    cd splits
    split -a 4 -d -n l/$n_jobs_plda $plda_feats_scp plda_feats.
    cd ../../

    extract_command="python $SCRIPT_DIR/extract_xvectors_gen_kaldi_input.py --out_dir=$output_dir $plda_vad_string --model=$model --scp=$job_dir/splits/plda_feats.\${ID} --window_size=1000000 --shift=1000000 --n_cores=1 --extract=$var_to_extract $side_info_string --store_format=h5 --context=22 $option_string > $job_dir/logs/extract_xvectors_gen_kaldi_input.py.\${ID}.log 2>&1"
    
    cat ${conf_dir}/tf_extract.conf | sed "s:JOB_DIR:${job_dir}:" | sed "s:W_DIR:${w_dir}:" > $job_dir/qsub.sh
    echo  $extract_command >> $job_dir/qsub.sh
fi



if [ $n_jobs_dev_eval -le 0 ];then
    echo "Skipping Dev and eval"
else
    echo "Extract x-vectors for dev and eval data"
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
    	if [[ $set_w_side_info == "dev_eval" ]] || [[ $set_w_side_info == "all" ]];then
	    side_info_string=${side_info_string}:1
    	else
    	    side_info_string=${side_info_string}:0
    	fi	
    fi
      
    scp=$dev_eval_scp

    job_dir=`pwd`/sge_dev_eval

    rm -r $job_dir
    mkdir $job_dir
    cd $job_dir

    mkdir splits logs    
    cd splits
    split -a 4 -d -n l/$n_jobs_dev_eval $dev_eval_feats_scp dev_eval_feats.
    cd ../../

    extract_command="python $SCRIPT_DIR/extract_xvectors_gen_kaldi_input.py --out_dir=$output_dir $dev_eval_vad_string --model=$model --scp=$job_dir/splits/dev_eval_feats.\${ID} --window_size=1000000 --shift=1000000 --n_cores=1 --extract=$var_to_extract $side_info_string --store_format=h5 --context=22 $option_string > $job_dir/logs/extract_xvectors_gen_kaldi_input.py.\${ID}.log 2>&1"
    
    cat ${conf_dir}/tf_extract.conf | sed "s:JOB_DIR:${job_dir}:" | sed "s:W_DIR:${w_dir}:" > $job_dir/qsub.sh
    echo  $extract_command >> $job_dir/qsub.sh
fi



if [ "A" == "A" ];then
    # "sync" prevents process from exit. "&" allows next jobs to start while first is
    # running. "wait" makes sure both of them has finished before next step starts.
    # (Removing "sync" would make the next step start directly even if wait is there)
    if [ $n_jobs_dev_eval -ge 1 ];then
	qsub -t 1-$n_jobs_dev_eval -sync yes sge_dev_eval/qsub.sh &
    fi
    if [ $n_jobs_plda -ge 1 ];then
	qsub -t 1-$n_jobs_plda -sync yes sge_plda/qsub.sh &
    fi
    wait
    echo "Extraction finished at " `date`
    
    mkdir ${tmp_dir}/models ${tmp_dir}/results

    if [ $n_jobs_dev_eval -ge 1 ];then
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
    fi

    if [ $n_jobs_plda -ge 1 ];then
	collect_embd_plda_command="python3 ${SCRIPT_DIR}/collect_embds_h5.py -d ${output_dir}/ -n plda_feats -of kaldi -s plda.scp"
	echo ${collect_embd_plda_command}
	eval ${collect_embd_plda_command} > collect_embd_plda.log 2>&1  
    fi
fi



