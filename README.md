# SEQUENS

This repository contains various code for speaker recogntion. The work started with Johan Rohdin's
post-doc project "SEQUENce Summarizing neural networks for speaker recognitions, SEQUENS, at Brno 
University of Technology 2016-2019. This code is part of the project that has received funding from 
the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie
and it is co-financed by the South Moravian Region under grant agreement No. 665860.
Sevaral members at Speech@BUT and also Themos Stafylakis at Omilia has contributed to this work. 


Distributed under the Apache license, version 2.0.


Notes:
 * The code in py3/tensorflow_code,the scripts "scripts/train_xvector_extractor_*" and "scripts/extract_xvectors_gen_kaldi_input.py" 
   uses tensorflow. The rest of the code, e.g., things for minibatch creation,  does should not be dependent on any toolkit.
 * The code Relies on Kaldi ( http://kaldi-asr.org/ ) for data preparation and
   https://github.com/gulamungon/kaldi-io-for-python/tree/read_feature_chunk for reading data. The latter is forked from
   https://github.com/vesis84/kaldi-io-for-python    
 * Contrary to Kaldi and most other toolkits, we don't prepare minibatches in advance and put them in archives. Instead,
   we create minibatches on-the-fly and read the features at that times. THIS ONLY WORKS IF FEATUREAS ARE STORED ON SSD.
 * Most code supports having segments of different length within a minibatch. This can be useful for end-to-end style training
   when segments are compared with eachother since we may want the system to work well also when comparing segments of different
   length. However, when the training objective is multiclass classification, it probably doesn't help. Note also that batch-norm
   (currently) can't be used if segments within the batch are of different duration.

Example usage

Training
1 Copy py3/scripts/examples/run_train_with_sge.sh to some directory where you want to run the training.  
2 Edit it to fit your environment.  
3 Run it --- it create a file callded qsub.sh  
4 Run: qsub qsub.sh  

Testing
1 Copy py3/scripts/examples/run_eval_with_sge.sh  
2 Edit it to fit your environmet. You need to creat a file some_path/conf/tf_extract.conf which contains  
  the first part of the qsub file to be used, for example something like  

\#!/bin/bash  
\#  
\#$ -cwd  
\#$ -S /bin/bash  
\#$ -N extr_xvec  
\#$ -o JOB_DIR/logs/extr_xvec.out  
\#$ -e JOB_DIR/logs/extr_xvec.err  
\#$ -q johan_extr  
cd W_DIR  
source some_path/venvs/py3.6_tf_20191108/bin/activate  
export PYTHONPATH=:some_path/software/kaldi-io-for-python:some_path/SEQUENS/py3//:some_path/SEQUENS/py3//  
export PATH=${PATH}:some_path/kaldi/src/featbin/:some_path/kaldi/src/ivectorbin/  
which python  
ID=`echo $SGE_TASK_ID | awk '{printf("%04d", $1-1) }'`

3 Run it. It will submit things to SGE by it self.  