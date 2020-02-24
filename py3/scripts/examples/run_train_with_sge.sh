#!/bin/bash

train_script="train_xvector_extractor_reco_loss_kaldi_feats.py" 
SEQUENS_py3_dir="some_path/SEQUENS/py3/"
kaldi_io_for_python="/workspace/jrohdin/software/kaldi-io-for-python"

if [ ! -e $train_script ];then
    cp $SEQUENS_py3_dir/$train_script .
fi

cat <<EOF > qsub.sh
#!/bin/bash
#
#$ -cwd
#$ -S /bin/bash
#$ -N train_xvec
#$ -o train_xvec.out
#$ -e train_xvec.err
#$ -q johan
cd `pwd`
source some_path/py3.6_tf/bin/activate
export PYTHONPATH=$PYTHONPATH:${kaldi-io-for-python}:${SEQUENS_py3_dir}/:${SEQUENS_py3_dir}/
which python
python $train_script > train_xvector_extractor_reco_kaldi_feats.log 2>&1
EOF
