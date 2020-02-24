#!/bin/bash

mkdir -p plda

data_dir=$1
lda_set=$2
lda_u2s=$3
plda_set=$4
plda_s2u=$5
adp_set=$6
evl_cnd=$7
trial_list_dir=$8
lda_dim=$9

if [ $# -eq 8 ] ;then
    echo "ERROR: the number of input arguments should be 9"
    exit -1
fi 

if [ "A" == "A" ];then
    # Compute the mean vector for centering the evaluation xvectors.
    if [ $adp_set != "None" ];then
	$train_cmd plda/compute_mean.log \
		   ivector-mean ark:${data_dir}/${adp_set}.ark \
		   plda/mean.vec || exit 1;
    else
	echo "WARNING: No adaptation set provided. PLDA set mean will be used for mean subtraction."
	$train_cmd plda/compute_mean.log \
		   ivector-mean ark:${data_dir}/${plda_set}.ark \
		   plda/mean.vec || exit 1;	
    fi

    # This script uses LDA to decrease the dimensionality prior to PLDA.
    $train_cmd plda/lda.log \
	       ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
	       "ark:ivector-subtract-global-mean ark:${data_dir}/${lda_set}.ark ark:- |" \
	       ark:${lda_u2s} plda/transform.mat || exit 1;

    # Train an out-of-domain PLDA model.
    $train_cmd plda/plda.log \
	       ivector-compute-plda ark:${plda_s2u} \
	       "ark:ivector-subtract-global-mean ark:${data_dir}/${plda_set}.ark  ark:- | transform-vec plda/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
	       plda/plda || exit 1;
    
    # Here we adapt the out-of-domain PLDA model.
    if [ $adp_set != "None" ];then
	$train_cmd plda/plda_adapt.log \
		   ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
		   plda/plda \
		   "ark:ivector-subtract-global-mean ark:${data_dir}/${adp_set}.ark ark:- | transform-vec plda/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
		   plda/plda_adapt || exit 1;
    fi
fi


# SRE16 trials
res=""
cond=""
info=""
for c in $(cat $evl_cnd);do 
    cond="$cond $c"
    info="$info eer minDCF0.01 minDCF0.001 #trials"
    
    e="output/${c}.enroll.ark"
    t="output/${c}.test.ark"

    trials="${trial_list_dir}/${c}.trial"
    cond="$cond\t$(wc -l $trials | cut -f1 -d' ' )"

    $train_cmd plda/scoring.log \
	       ivector-plda-scoring --normalize-length=true \
	       "ivector-copy-plda --smoothing=0.0 plda/plda - |" \
	       "ark:ivector-subtract-global-mean plda/mean.vec ark:${e} ark:- | transform-vec plda/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
	       "ark:ivector-subtract-global-mean plda/mean.vec ark:${t} ark:- | transform-vec plda/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" "cat '$trials' | cut -d\  --fields=1,2 |" plda/scores_${c} || exit 1;
    awk 'NR==FNR{a[$1$2]=$3;next}{print $3 " " a[$1$2]}' $trials plda/scores_${c} > plda/scores_lab_${c} 
    res="$res $(cat plda/scores_lab_${c}  | compute-eer - 2> /dev/null )"
    res="$res $(compute_min_dcf.py --p-target 0.01 plda/scores_${c} $trials 2> /dev/null)"
    res="$res $(compute_min_dcf.py --p-target 0.001 plda/scores_${c} $trials 2> /dev/null)"
    res="$res $(wc -l plda/scores_lab_${c} | cut -f1 -d' ' )"
done
echo -e $cond
echo -e $info
echo -e $res 




