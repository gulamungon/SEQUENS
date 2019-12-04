#!/bin/bash


#feat_dir=/mnt/matylda6/rohdin/expts/runs/x-vec_sitw_base_expts/feats_sitw_xvec
#feat_dir_voxceleb=/mnt/matylda6/rohdin/expts/runs/x-vec_voxceleb1_base_expts/data_prep/mfcc/
#vad_dir=/mnt/matylda6/rohdin/expts/runs/x-vec_sitw_base_expts/vad/

w_dir=`pwd`

# Nasty way to make temporary storing at scratch
out=`echo  $w_dir | sed "s:.*/mnt/matylda6/rohdin/expts/runs/x-vec_python_expts::"`
out="/mnt/scratch06/tmp/rohdin/${out}/"

model_dir=$1 
model=$2
var_to_extract=$3
SCRIPT_DIR=`readlink -f $0 | xargs dirname`  # The directory of this script 


if [ $# -ne 3 ] && [ $# -ne 5 ];then
   echo "Usage:    $0 model_dir model_file var_to_extract [side_info]"
   echo "Example 1: $0 /mnt/matylda6/rohdin/expts/runs/x-vec_python_train/test_61_cleanup_cont/output/ model_feat2score_epoch-9_lr-0.001_lossTr-0.167914892249_lossDev-0.207157.h5 embd_A,embd_B"
   echo "Example 2: $0 /mnt/matylda6/rohdin/expts/runs/x-vec_python_train/test_61_cleanup_cont/output/ best (Model with best dev loss is used) embd_A,embd_B"
   exit
fi

if [ $# -eq 5 ] ;then
    side_info=$4
    echo "Using $side_info as side info to TDNN."
    side_info_string="--side_info=$side_info"
    set_w_side_info=$5
    echo "Adding side_info value 1 to $5"
    
else
    echo "No side info is used in  TDNN."
    side_info="None"
    side_info_string=""
    side_info_value=""
fi


model=$model_dir/$model
echo "Model: $model"


# Make a qsub template. 
function mk_qsub_top {
    jobdir=$1
    outfile=$jobdir/qsub.sh
    cat << EOF > $outfile
#!/bin/bash
#
#$ -cwd
#$ -V
#$ -N extr_xvec
#$ -o $job_dir/logs/extr_xvec.out
#$ -e $job_dir/logs/extr_xvec.err
#$ -l ram_free=4G,mem_free=4G,matylda6=0.5
#$ -q all.q@blade065,all.q@blade066,all.q@blade067,all.q@blade068,all.q@blade069,all.q@blade070,all.q@blade071,all.q@blade072,all.q@blade073,all.q@blade074,all.q@blade075,all.q@blade076,all.q@blade077,all.q@blade078,all.q@blade079,all.q@blade080,all.q@blade081,all.q@blade082,all.q@blade083,all.q@blade084,all.q@blade085,all.q@blade086,all.q@blade087,all.q@blade088,all.q@blade089,all.q@blade090,all.q@blade091,all.q@blade092,all.q@blade093,all.q@blade094,all.q@blade095,all.q@blade096,all.q@blade097,all.q@blade098,all.q@blade099,all.q@blade100,all.q@blade101,all.q@blade102
cd `pwd`
unset PYTHONPATH
unset PYTHONHOME
export PATH="/homes/kazi/rohdin/.conda/envs/anaconda3_lab_20190806/bin:$PATH"
export PYTHONPATH=$PYTHONPATH
which python
EOF
    echo "ID=\`echo \$SGE_TASK_ID | awk '{printf(\"%04d\", \$1-1) }'\`" >> $outfile
}


# Extract x-vectors for plda train data
if [ "A" == "B" ];then

    plda_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/plda_train/feats.scp"
    vad_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/plda_train/vad.scp"
    
    if [ $side_info != "None" ];then
    	if [ $set_w_side_info == "plda_train" ];then
    	    side_info_value=1
    	else
    	    side_info_value=0
    	fi	
    fi
   
    output_dir=${out}/output
    mkdir -p $output_dir
    #for dir in `cut -f2 -d"="  $plda_scp | xargs dirname | sort -u`;do
    #	mkdir -p $output_dir/$dir
    #done
   
    scp=$plda_scp

    job_dir=`pwd`/sge_plda_train
    n_splits_plda=6000

    rm -r $job_dir
    mkdir $job_dir
    cd $job_dir

    mkdir splits logs    
    cd splits
    split -a 4 -d -n l/$n_splits_plda $scp plda.
    cd ../../

    mk_qsub_top $job_dir 
    echo "python $SCRIPT_DIR/extract_xvectors_gen_kaldi_input.py --out_dir=$output_dir --vad_scp=$vad_scp --model=$model --scp=$job_dir/splits/plda.\${ID} --window_size=1000000 --shift=1000000 --n_cores=1 --extract=$var_to_extract $side_info_string:$side_info_value --store_format=h5 --context=22 > $job_dir/logs/feat_2_xvec_tf.py.\${ID}.log 2>&1" >> $job_dir/qsub.sh
fi

# Extract x-vectors for SRE18
if [ "A" == "A" ];then

    scp=`pwd`/sre18_dev_eval.scp
    vad_scp=`pwd`/sre18_dev_eval_vad.scp

    dev_test_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_dev_cmn2.test/feats.scp"
    dev_enroll_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_dev_cmn2.enroll/feats.scp"
    eval_test_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_evl_cmn2.test/feats.scp"
    eval_enroll_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_evl_cmn2.enroll/feats.scp"
    dev_unlab_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_dev_unlabeled/feats.scp"
    #evl_trn_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_cmn2_eval_trn_with_aug/feats.scp"
    
    dev_test_vad_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_dev_cmn2.test/vad.scp"
    dev_enroll_vad_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_dev_cmn2.enroll/vad.scp"
    eval_test_vad_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_evl_cmn2.test/vad.scp"
    eval_enroll_vad_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_evl_cmn2.enroll/vad.scp"
    dev_unlab_vad_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_dev_unlabeled/vad.scp"
    #evl_trn_vad_scp="/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/cmn2_2019_fbank/data/sre18_cmn2_eval_trn_with_aug/vad.scp"
    
    cat $dev_test_vad_scp $dev_enroll_vad_scp $eval_test_vad_scp $eval_enroll_vad_scp $dev_unlab_vad_scp $evl_trn_vad_scp  > $vad_scp
    cat $dev_test_scp $dev_enroll_scp $eval_test_scp $eval_enroll_scp $dev_unlab_scp $evl_trn_scp > $scp
    
    if [ $side_info != "None" ];then
    	if [ $set_w_side_info == "sre18" ];then
    	    side_info_value=0
    	else
    	    side_info_value=0
    	fi	
    fi
    
    sort -u $scp | shuf > ${scp}.unique
    scp=${scp}.unique
    
    output_dir=${out}/output
    mkdir -p $output_dir
    #for dir in `cut -f2 -d"="  $scp | xargs dirname | sort -u`;do
    #	mkdir -p $output_dir/$dir
    #done

    job_dir=`pwd`/sge_sre18
    n_splits_sre18=350

    rm -r $job_dir
    mkdir $job_dir
    cd $job_dir
    mkdir splits logs    
    cd splits
    split -a 4 -d -n l/$n_splits_sre18 $scp sre18.
    cd ../../
    
    mk_qsub_top $job_dir 
    echo "python $SCRIPT_DIR/extract_xvectors_gen_kaldi_input.py --out_dir=$output_dir --vad_scp=$vad_scp --model=$model --scp=$job_dir/splits/sre18.\${ID} --window_size=1000000 --shift=1000000 --n_cores=1 --extract=$var_to_extract $side_info_string:$side_info_value --store_format=h5 --context=22 > $job_dir/logs/feat_2_xvec_tf.py.\${ID}.log 2>&1" >> $job_dir/qsub.sh
fi



if [ "A" == "A" ];then
    qsub -t 1-$n_splits_sre18 sge_sre18/qsub.sh
    #qsub -t 1-$n_splits_plda -sync yes sge_plda_train/qsub.sh
    #mkdir ${out}/models ${out}/results

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



