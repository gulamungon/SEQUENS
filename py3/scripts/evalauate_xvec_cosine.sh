#!/bin/bash

mkdir -p cosine

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
	$train_cmd cosine/compute_mean.log \
		   ivector-mean ark:${data_dir}/${adp_set}.ark \
		   cosine/mean.vec || exit 1;
    else
	echo "WARNING: No adaptation set provided. PLDA set mean will be used for mean subtraction."
	$train_cmd cosine/compute_mean.log \
		   ivector-mean "ark:ivector-normalize-length ark:${data_dir}/${plda_set}.ark ark:- |" \
		   cosine/mean.vec || exit 1;	
    fi

    # This script uses LDA to decrease the dimensionality prior to PLDA.
    #$train_cmd plda/lda.log \
#	       ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
#	       "ark:ivector-normalize-length ark:${data_dir}/${lda_set}.ark ark:- | ivector-subtract-global-mean ark:- ark:- |" \
#	       ark:${lda_u2s} plda/transform.mat || exit 1;

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

    #$train_cmd cosine/scoring.log \
    #	       ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    #	       "ark:ivector-normalize-length ark:${e} ark:- | ivector-subtract-global-mean cosine/mean.vec ark:- ark:- | transform-vec plda/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    #	       "ark:ivector-normalize-length ark:${t} ark:- | ivector-subtract-global-mean cosine/mean.vec ark:- ark:- | transform-vec plda/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"  cosine/scores_${c} || exit 1;
    #awk 'NR==FNR{a[$1$2]=$3;next}{print $3 " " a[$1$2]}' $trials cosine/scores_${c} > cosine/scores_lab_${c} 
    #res="$res $(cat cosine/scores_lab_${c}  | compute-eer - 2> /dev/null )"
    #res="$res $(/workspace/jrohdin/kaldi/egs/voxceleb/v2/sid/compute_min_dcf.py --p-target 0.01 cosine/scores_${c} $trials 2> /dev/null)"
    #res="$res $(/workspace/jrohdin/kaldi/egs/voxceleb/v2/sid/compute_min_dcf.py --p-target 0.001 cosine/scores_${c} $trials 2> /dev/null)"
    #res="$res $(wc -l cosine/scores_lab_${c} | cut -f1 -d' ' )"
    
    $train_cmd cosine/scoring.log \
    	       ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    	       "ark:ivector-normalize-length ark:${e} ark:- |" \
    	       "ark:ivector-normalize-length ark:${t} ark:- |"  cosine/scores_${c} || exit 1;
    awk 'NR==FNR{a[$1$2]=$3;next}{print $3 " " a[$1$2]}' $trials cosine/scores_${c} > cosine/scores_lab_${c} 
    res="$res $(cat cosine/scores_lab_${c}  | compute-eer - 2> /dev/null )"
    res="$res $(/workspace/jrohdin/kaldi/egs/voxceleb/v2/sid/compute_min_dcf.py --p-target 0.01 cosine/scores_${c} $trials 2> /dev/null)"
    res="$res $(/workspace/jrohdin/kaldi/egs/voxceleb/v2/sid/compute_min_dcf.py --p-target 0.001 cosine/scores_${c} $trials 2> /dev/null)"
    res="$res $(wc -l cosine/scores_lab_${c} | cut -f1 -d' ' )"

    #ch="output/plda.ark"

    #trials="/workspace/jrohdin/expts/tf_xvector_baselines/lists/voxceleb/trials/${c}.enroll_z_cohort.trials"
    #$train_cmd cosine/scoring_z_cohort.log \
    #	       ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |"  \
    #	       "ark:ivector-normalize-length ark:${e} ark:- |" \
    #	       "ark:ivector-normalize-length ark:${ch} ark:- |"  cosine/scores_z_cohort_${c} || exit 1;

    #trials="/workspace/jrohdin/expts/tf_xvector_baselines/lists/voxceleb/trials/${c}.test_t_cohort.trials"
    #$train_cmd cosine/scoring_t_cohort.log \
    #	       ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    #	       "ark:ivector-normalize-length ark:${t} ark:- |" \
    #	       "ark:ivector-normalize-length ark:${ch} ark:- |"  cosine/scores_t_cohort_${c} || exit 1;

    
    #/workspace/petr/gitlab/evalTools_kaldi554/sv/score-norm/omilia-bio-score-norm --score-file=cosine/scores_${c} --z-score-file=cosine/scores_z_cohort_${c} --t-score-file=cosine/scores_t_cohort_${c} --top-percent=10 > cosine/scores_norm_${c}

    
    #awk 'NR==FNR{a[$1$2]=$3;next}{print $3 " " a[$1$2]}' $trials cosine/scores_norm_${c} > cosine/scores_lab_norm_${c} 
    #res="$res $(cat cosine/scores_lab_norm_${c}  | compute-eer - 2> /dev/null )"
    #res="$res $(/workspace/jrohdin/kaldi/egs/voxceleb/v2/sid/compute_min_dcf.py --p-target 0.01 cosine/scores_norm_${c} $trials 2> /dev/null)"
    #res="$res $(/workspace/jrohdin/kaldi/egs/voxceleb/v2/sid/compute_min_dcf.py --p-target 0.001 cosine/scores_norm_${c} $trials 2> /dev/null)"
    #res="$res $(wc -l cosine/scores_lab_norm_${c} | cut -f1 -d' ' )"

done
echo -e $cond
echo -e $info
echo -e $res 


#$train_cmd cosine/scoring.log \
#	   ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
#	   "ark:ivector-subtract-global-mean plda/mean.vec ark:${e} ark:- | ivector-normalize-length ark:- ark:- |" \
#	   "ark:ivector-subtract-global-mean plda/mean.vec ark:${t} ark:- | ivector-normalize-length ark:- ark:- |"  cosine/scores_${c} || exit 1;
#awk 'NR==FNR{a[$1$2]=$3;next}{print $3 " " a[$1$2]}' $trials cosine/scores_${c} > cosine/scores_lab_${c} 


