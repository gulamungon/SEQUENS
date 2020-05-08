#!/bin/bash

mkdir -p cosine

data_dir=$1
mean_set=$2
lda_set=$3
lda_u2s=$4
evl_cnd=$5
trial_list_dir=$6
lda_dim=$7
out_dir=$8

if [ $# -eq 6 ] ;then
    echo "ERROR: the number of input arguments should be 9"
    exit -1
fi 

if [ "A" == "A" ];then
    # Compute the mean vector for centering the evaluation xvectors.
    if [ $mean_set != "None" ];then
	$train_cmd $out_dir/compute_mean.log \
		   ivector-mean ark:${data_dir}/${mean_set}.ark \
		   $out_dir/mean.vec || exit 1;
    elif [ $lda_set != "None" ];then
	$train_cmd $out_dir/compute_mean.log \
		   ivector-mean ark:${data_dir}/${lda_set}.ark \
		   $out_dir/mean.vec || exit 1;
    else    
	echo "WARNING: No mean set provided. No mean will be subtraction."
    fi
    
    if [ $lda_set != "None" ];then	
	$train_cmd plda/lda.log \
		   ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
		   "ark:ivector-subtract-global-mean ark:${data_dir}/${lda_set}.ark ark:- |" \
		   ark:${lda_u2s} $out_dir/transform.mat || exit 1;	
    else
	echo "WARNING: No LDA set provided. LDA will not be applied."
    fi

fi


# Trials
res=""
cond=""
info=""
for c in $(cat $evl_cnd);do 
    cond="$cond $c"
    info="$info eer minDCF0.01 minDCF0.001 #trials"
    
    #e="output/${c}.enroll.ark"
    #t="output/${c}.test.ark"

    trials="${trial_list_dir}/${c}.trial"
    cond="$cond\t$(wc -l $trials | cut -f1 -d' ' )"

    if [ $lda_set != "None" ];then
	feats_e="ark:ivector-subtract-global-mean $out_dir/mean.vec ark:output/${c}.enroll.ark ark:- | transform-vec plda/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
	feats_t="ark:ivector-subtract-global-mean $out_dir/mean.vec ark:output/${c}.test.ark ark:- | transform-vec plda/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
    elif [ $mean_set != "None" ];then
	feats_e="ark:ivector-subtract-global-mean $out_dir/mean.vec ark:output/${c}.enroll.ark ark:- | ivector-normalize-length ark:- ark:- |"
	feats_t="ark:ivector-subtract-global-mean $out_dir/mean.vec ark:output/${c}.test.ark ark:- | ivector-normalize-length ark:- ark:- |"
    else
	feats_e="ark:ivector-normalize-length ark:output/${c}.enroll.ark ark:- |"
	feats_t="ark:ivector-normalize-length ark:output/${c}.test.ark ark:- |"
    fi
	
    $train_cmd $out_dir/scoring.log \
    	       ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    	       "$feats_e" "$feats_t" $out_dir/scores_${c} || exit 1;
    
    awk 'NR==FNR{a[$1$2]=$3;next}{print $3 " " a[$1$2]}' $trials $out_dir/scores_${c} > $out_dir/scores_lab_${c} 
    res="$res $(cat $out_dir/scores_lab_${c}  | compute-eer - 2> /dev/null )"
    res="$res $(compute_min_dcf.py --p-target 0.01 $out_dir/scores_${c} $trials 2> /dev/null)"
    res="$res $(compute_min_dcf.py --p-target 0.001 $out_dir/scores_${c} $trials 2> /dev/null)"
    res="$res $(wc -l $out_dir/scores_lab_${c} | cut -f1 -d' ' )"

done
echo -e $cond
echo -e $info
echo -e $res 




