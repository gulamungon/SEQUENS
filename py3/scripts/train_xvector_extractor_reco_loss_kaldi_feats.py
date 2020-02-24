#!/usr/bin/env python

### This script is for training Kaldi style x-vectors with Tensorflow. 

### NOTES
# * A suffix "_" denotes a TF symbolic variable and "_p" denotes a TF placeholder variable
# * The model components, e.g., TDNNs or feed-forward NNs are classes, similar to Keras.
#   This was to make a neat interface, in particular for managing segments of different
#   duration within the batch (see next comment.)
# * The strategy for allowing segments within a batch to be of different duration is a bit
#   complicated to follow. In order to use segment of different lengths withing the minibatch,
#   all segments are first concatenated to one long segment and a array of indices where
#   the segments starts and ends is produced. Then we are looping over the indices by the
#   Tensorflow while loop and in processing the segments one by one until the pooling stage.
#   This is done by the class "tensorflow_code.nn_def.tf_pool". After pooling we have
#   fixed length embeddings for each segment so the processing can continue in a more straigh-
#   forward way. The class "tensorflow_code.nn_def.tf_pool" requires an input feature variable,
#   a corresponding indices variable and a list of NN components that the features should be
#   sent through, as well as a pooling function as input. See the code for details.

# Todo: 
# * Fix so that variable duration segments can be used for reconstruction dev. loss
# * Fix so that segment for embedding and reconstruction loss never overlaps? Now both of them are randomly selected.
#

floatX='float32'


import sys, os, pickle, copy, subprocess, time, inspect, re #, h5py 

from utils.misc import get_logger
log = get_logger()

    
from tensorflow_code import pool_fkns

import numpy as np
import utils.mbatch_generation
from utils.load_data_kaldi import *
from utils.model_utils import  load_davids_kaldi_model
import tensorflow_code.optim
import tensorflow_code.models
import tensorflow as tf
from utils.train import train_nn, get_train_batch_fkn  
from tensorflow_code.dplda import mBW_2_PQck
from tensorflow_code.dplda import p_eff, llrThreshold, labMat2weight, lab2matrix
from tensorflow_code.load_save import save_tf_model
from tensorflow_code.nn_def import tf_ff_nn

from utils.misc import make_dir_mpi
from utils.misc import extract_embeddings

from utils.model_utils import load_model
import scipy.linalg as spl

from tensorflow_code.dplda import dplda_simple
from tensorflow_code.dplda import mBW_2_PQck

import kaldi_io

if ( __name__ == "__main__" ):

    
    # Check hostname and cpu info. Will be printed in log below. Cpu info just checks
    # the first cpu on the machine, not necessarily the one we use but normally they
    # are the same.
    host_name  = os.uname()[1]
    cpu_info   = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | head -n 1 ", shell=True).decode('utf-8').split(':')[1]

    # To make it save at the right place in interactive use. 
    if os.path.basename(os.getcwd()) == 'scripts':
        work_dir   = os.getcwd() + '/../../../'
    else:
        work_dir  =os.getcwd()

    output_dir = work_dir + "/output/"
    log_dir    = work_dir + "/logs/"
    log.info("Host: " + host_name)
    log.info("CPU info:" + cpu_info)
    log.info("Work dir: " + work_dir)

    if not os.path.exists( output_dir ):
        os.makedirs( output_dir )
    if not os.path.exists( log_dir ):
        os.makedirs( log_dir )

        
    # Output model
    model_prefix = output_dir + '/model'

### --- Data etc. ------------------------------------------ ###

    # In this directory there should be one file for each utt_id named "utt_id.hp", e.g., "id00012-21Uxsk56VDQ-00001.hp"
    # The content should be one integer per row that indicates the phoneme label of the frame, i.e., there should be one row per frame (before VAD)
    phn_lab_dir = '/workspace/jrohdin/expts/asr/train_fskip/decode_rnnlm_lstm_tdnn_a_averaged/9_phone_post/'

    # The scp files below should have the format "spk_id utt_id ark num_frames" 
    # id08343 id08343-GjRtqOHJHDA-00155 some_path/kaldi/egs/voxceleb/v2/exp/train_combined_no_sil/xvector_feats_train_combined.58.ark:491844037 723
    # It is assumed that VAD has already been applied to the utterances
    train_scp   = "train.scp"
    dev_scp     = "dev_multi.scp"  # Used to check speaker classification accuracy during training. The speaker should overlap with the training set but utterance should not.
    
    # VAD scp in Kaldi format. It is used to remove non-speech frames from the ASR output
    vad_scp     = "some_path/kaldi/egs/voxceleb/v2/data/train_combined/vad.scp" 

    # Data for verification check during training. The speakers in this data should not overlap with the training set.
    # The format is the standard Kaldi format.
    dev_trial_scp = "dev_ver.2K.scp"      
    dev_trial_lst = "dev_ver.2K.trials"

    train_scp_info_file     = output_dir +'train_scp_info.pkl'
    dev_scp_info_file       = output_dir +'dev_scp_info.pkl'
    dev_trial_scp_info_file = output_dir +'dev_trial_scp_info.pkl'

    
    # File length limits including context
    min_length   = 200
    max_length   = 400
    min_length_2 = 200  
    max_length_2 = 400  
    #min_length        = 300
    #max_length        = 301 

    segm_B_clean = True
    
### --- Training settings ------------------------------- ###
        
    annoying_train = True # This forces all segments within a batch to be of the same length. This is necessary in order for bathchorm to work.

    bn_decay   = 0.95 # Decay of moving average in batchnorm (for estimating stats used in testing).     

    n_utt_per_spk = 1
    batchsize  = 150
    n_epoch    = 5000            # Maximum number of epochs
    n_batch    = 400             # Number of minimbatches per epoch (dev loss is checked after each epoch) 
    lr_first   = 1e-2            # Initial learning rate. Will be halved if there is no improvent on dev loss
    lr_last    = 1e-6            # Last learning rate. When we have reduced the learning rate below this value  
                                 # the training stops. See the training scheme below.

    patience  = 31               # For one learning rate, the training is allowed to fail this many times.
                                 # If it fails one more time, the learning rate will be halved, parameters
                                 # reset and the training continued.
                                 
    patience_2 = -1              # Use a different patience after 150 epochs.
                                 
    batch_que_length = 2         # Number of batch in que (will be prepared in parallel with training)    
    half_every_N_epochs = -1
    
    lr_p, optim, get_optim_states = tensorflow_code.optim.get_optimizer('Adam', floatX=floatX)
    log.info("Optimizer: " + optim._name)

    kaldi_txt_model = None
    tf_model    = None
    lr_first    = 0.01
    seed_offset = 0 * 1357 
    assert( (kaldi_txt_model == None) or (tf_model == None) )

    spk_class_train_only = True
    reco_train_only     = False
    assert ( (not spk_class_train_only) or (not reco_train_only) )

    stop_grad_ap = -1 # -1 Doesn't stopp it. 2 Stops it between first TDNN and first Relu after pooling (Other values would stop the gradient at other places)
    stop_reco_grad = False
    if spk_class_train_only:
        log.info("Using only speaker classification objective for training x-vector")
        stop_reco_grad = True
        ret_loss = "loss_c"
    elif reco_train_only:
        stop_grad_ap = 2
        log.info("Using only reconstruction objective for training x-vector")
        ret_loss = "loss_reco"
    else:
        log.info("Using both speaker classification and reconstruction objective for training x-vector")
        ret_loss = "Loss_1"
#############################################################################            
### --- Load and prepare data ------------------------------------------- ###
#############################################################################    

    p_aug_remove = re.compile('-babble|-noise|-music|-reverb')
    # Create functions for loading data given a set of files. 
    if annoying_train:
        # All segments will be of the same length
        rng_f_tr = np.random.RandomState( seed = 519 + seed_offset )
        def load_feats_train(u):
            start_from_zero = False
            n_avl_samp     = np.array( [train_scp_info['utt2sideInfo'][uu] for uu in u] ) 
            files          = [train_scp_info['utt2file'][uu] for uu in u]
            return  load_kaldi_feats_segm_same_dur(rng_f_tr, files, min_length, max_length, n_avl_samp, start_from_zero=False)
        
        rng_f_phn_tr_A = np.random.RandomState( seed = 534 + seed_offset )
        rng_f_phn_tr_B = np.random.RandomState( seed = 120 + seed_offset )
        def load_feats_train_joint(u):
            start_from_zero = False
            n_avl_samp      = np.array( [train_scp_info['utt2sideInfo'][uu] for uu in u] ) 
            files           = [train_scp_info['utt2file'][uu] for uu in u]
            f_ids           = [p_aug_remove.sub('', train_scp_info['utt2fid'][uu]) for uu in u]       # File IDs
            vad_files       = [vad_fid2file[i] for i in f_ids ]
            data_A          = load_kaldi_feats_segm_same_dur(rng_f_phn_tr_A, files, min_length, max_length, n_avl_samp, start_from_zero=False)

            if segm_B_clean:
               files = [ train_scp_info['fid2file'][x] for x in f_ids ]
            
            data_B, lab     = load_kaldi_feats_segm_same_dur_plus_lab(rng_f_phn_tr_B, files, min_length, max_length, n_avl_samp, phn_lab_dir, f_ids, vad_files, start_from_zero)
            return data_A, data_B, lab
    else:
        # Segments can be of different lenght
        log.error("Not supported.")
        os.sys.exit()


    def load_feats_dev_joint(u):
        start_from_zero = False
        n_avl_samp      = np.array( [dev_scp_info['utt2sideInfo'][uu] for uu in u] ) 
        files           = [dev_scp_info['utt2file'][uu] for uu in u]
        f_ids           = [p_aug_remove.sub('', dev_scp_info['utt2fid'][uu]) for uu in u]       # File IDs
        vad_files       = [vad_fid2file[i] for i in f_ids ]
        rng_f_phn_de_A    = np.random.RandomState(seed=188)    # Set the seed here to make sure same segments are loaded each time.
        rng_f_phn_de_B    = np.random.RandomState(seed=10091)
        data_A          = load_kaldi_feats_segm_same_dur(rng_f_phn_de_A, files, min_length=300, max_length=301, n_avl_samp=n_avl_samp, start_from_zero=False)
        C               = np.arange(0, (data_A.shape[0]+1)*data_A.shape[1], data_A.shape[1]  )
        data_A          = data_A.reshape(1,-1, data_A.shape[2])

        if segm_B_clean:
            files = [ dev_scp_info['fid2file'][x] for x in f_ids ]

        data_B, lab     = load_kaldi_feats_segm_same_dur_plus_lab(rng_f_phn_de_B, files, 300, 301, n_avl_samp, phn_lab_dir, f_ids, vad_files, start_from_zero)
        print (data_A.shape)
        print (data_B.shape)
        return data_A, C, data_B, lab
        
    rng_f_phn_or    = np.random.RandomState(seed=11681)  
    def load_feats_train_only_reco(u):
        
        n_avl_samp      = np.array( [train_scp_info['utt2sideInfo'][uu] for uu in u] ) 
        files           = [train_scp_info['utt2file'][uu] for uu in u]
        f_ids           = [ p_aug_remove.sub('', train_scp_info['utt2fid'][uu])  for uu in u]       # File IDs
        vad_files       = [vad_fid2file[i] for i in f_ids ]

        if segm_B_clean:
            files = [ train_scp_info['fid2file'][x] for x in f_ids ]
            
        data, lab       = load_kaldi_feats_segm_same_dur_plus_lab(rng_f_phn_or, files, min_length, max_length, n_avl_samp, phn_lab_dir, f_ids, vad_files, start_from_zero=False)
        return data, lab


    # Read in the VAD info
    f = open(vad_scp, 'r')
    vad_fid2file={}
    for line in f:
        u,f  = line.rstrip().split(" ")
        vad_fid2file[u]=f
        
    # Prepare train scp info
    if os.path.exists( train_scp_info_file ):

        log.info("Loading scp info from " + train_scp_info_file)
        inp            = open( train_scp_info_file, 'rb')
        train_scp_info = pickle.load(inp)
        inp.close()
        log.info("Number of speakers: %d, number of utterances: %d ", len(train_scp_info['spk_name']), len(train_scp_info['utt2spk'] ))
    else:
        train_scp_info = utils.mbatch_generation.get_kaldi_scp_info( train_scp )
        
        log.info( "Saving scp to " + train_scp_info_file )
        out = open(train_scp_info_file, 'wb')
        pickle.dump(train_scp_info, out)
        out.close()

        
    # Prepare dev (valid) scp info
    if os.path.exists( dev_scp_info_file ):

        log.info("Loading scp info from " + dev_scp_info_file)
        inp          = open( dev_scp_info_file, 'rb')
        dev_scp_info = pickle.load(inp)
        inp.close()
    else:
        dev_scp_info = utils.mbatch_generation.get_kaldi_scp_info( dev_scp, train_scp_info['spk_name'] )
        log.info( "Saving scp to " + dev_scp_info_file )
        out = open(dev_scp_info_file, 'wb')
        pickle.dump(dev_scp_info, out)
        out.close()

    n_spk     = train_scp_info['spk_counts'].shape[0]    

    print(n_spk)
    

    # Read in the dev trial files
    f = open(dev_trial_scp, 'r')
    dev_trial_files=[]
    dev_trial_id2ind={}
    for i,line in enumerate(f):
        idd,fl = line.rstrip().split(" ")
        dev_trial_files.append( fl )
        dev_trial_id2ind[idd] = i

    f = open(dev_trial_lst, 'r')
    dev_trial_enr  = []
    dev_trial_test = []
    dev_trial_lab = []
    for i,line in enumerate(f):
        e,t,l = line.rstrip().split(" ")
        dev_trial_enr.append( e ) 
        dev_trial_test.append( t ) 
        dev_trial_lab.append( l ) 
        
    def load_feats_dev_trial(files):
        d= [kaldi_io.read_mat(f) for f in files]
        idx = np.cumsum( np.vstack([ d[i].shape[0] for i in range(len(d)) ]  ) )
        idx = np.insert(idx, 0,0)
        d = np.vstack(d)[np.newaxis,:,:]
        return d,idx.astype(floatX)

    
    # Detect which GPU to use
    command='nvidia-smi --query-gpu=memory.free,memory.total --format=csv |tail -n+2| awk \'BEGIN{FS=" "}{if ($1/$3 > 0.98) print NR-1}\''
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = subprocess.check_output(command, shell=True).decode('utf-8').rsplit('\n')[0]
        log.info("CUDA_VISIBLE_DEVICES " + os.environ["CUDA_VISIBLE_DEVICES"])
    except subprocess.CalledProcessError:
        log.info("No GPU seems to be available")        
    sess            = tf.Session()

   
    ### ---- Data generators used for model initilalization. --- #
    # Generator for training will be defined later.
    rng_tr = np.random.RandomState(seed=123)    

    batch_count = 0
    it_tr=utils.mbatch_generation.gen_mbatch_spk_bal(train_scp_info, ivec_dir=None, stats_dir=None, feat_dir=None, stats_order=2,
                                                     frame_step=1, max_length=30000,  
                                                     y_function=None, verbose=False,
                                                     arrange_spk_fcn = None, n_spk_per_batch=batchsize, n_utt_per_spk=n_utt_per_spk,
                                                     output_labs=True, output_scp_ind=False, output_utt_id=True,
                                                     out2put_utt2sideInfo=False, rng=rng_tr )        

    # The batch_iterator class prepares batches in a second thread while the training is running.
    all_bad_utts = [] # Keeps track of 0-duration utterances. (Which are ignored in trainin). Not used if annoying_train=True  
    it_tr_que    = utils.mbatch_generation.batch_iterator_2(it_tr, load_feats_train,
                                                          annoying_train, batch_que_length, batch_number=0, use_mpi=False)

    
    rng_tr_phn = np.random.RandomState(seed=208)    
    it_tr_phn  = utils.mbatch_generation.gen_mbatch_spk_bal(train_scp_info, ivec_dir=None, stats_dir=None, feat_dir=None, stats_order=2,
                                                            frame_step=1, max_length=30000,  
                                                            y_function=None, verbose=False,
                                                            arrange_spk_fcn = None, n_spk_per_batch=batchsize, n_utt_per_spk=n_utt_per_spk,
                                                            output_labs=True, output_scp_ind=False, output_utt_id=True,
                                                            out2put_utt2sideInfo=False, rng=rng_tr_phn )        

    # The batch_iterator class prepares batches in a second thread while the training is running.
    all_bad_utts = [] # Keeps track of 0-duration utterances. (Which are ignored in trainin). Not used if annoying_train=True
    # Note that load feat dev 2 uses a fixed seed...
    it_tr_phn_que = utils.mbatch_generation.batch_iterator_2(it_tr_phn, load_feats_train_only_reco,
                                                             annoying_train, batch_que_length, batch_number=0, use_mpi=False)
    
    
#############################################################################    
### --- Set up the model ------------------------------------------------ ###
#############################################################################

    # Variables. 
    X1_p        = tf.placeholder(floatX, shape=[None,None,30], name='X1_p') # Features 
    C1_p        = tf.placeholder(dtype='int32', shape=[None], name='C1_p')  # Indices
    is_test_p   = tf.placeholder(dtype='bool', shape=[], name='is_test_p')  # Tells whether it is the training or the testing phase.
    WGT_c_p     = tf.placeholder(floatX, shape=[None], name='WGT_c_p')      # Weight matrix for class
    L_c_p       = tf.placeholder('int32', shape=[None], name='L_c_out_p')   # Label vector for class out domain class (will be hard)

    P2_p        = tf.placeholder('int32', shape=[None,None], name='P1_p')      # Hard posteriors (phonems etc)
    X2_r_p      = tf.placeholder(floatX, shape=[None,None,30], name='X2_r_p') # Features to reconstruct

    # We subtract 1 from P2_p since they are 1 one based. As an exception,
    # keep the _p suffix
    
    P2_p = P2_p -1 
    tdnn_sizes_before_pooling   = ( [150, 512, 5, 1], [1536, 512, 3, 2], [1536,512, 3, 3], [512, 512, 1,1], [512,1500,1,1] )
    tdnn_sizes_after_pooling    = ( [3000, 512, 1, 1], [512, 512, 1, 1] )
    activations_before_pooling  = [tf.nn.relu] * 5
    activations_after_pooling   = [tf.nn.relu] * 2
    pool_size                   = 3000
        
    do_feat_norm   = True
    do_pool_norm   = True

    # Which part of the network to update
    update_feat_norm   = False
    update_b_pool      = True
    update_pool_norm   = False
    update_a_pool      = True
    update_class       = True

    # Regularization. Used a bit further below in the code.
    l2_reg_b_pool = 0.0 
    l2_reg_a_pool = 0.0 
    l2_reg_class  = 0.0 
    
    if update_feat_norm and (not do_feat_norm):
        log.warning("WARNING: update_feat_norm will be changed to False because do_feat_norm = False")
        update_feat_norm   = False
    if update_pool_norm and (not do_pool_norm):
        log.warning("WARNING: update_pool_norm will be changed to False because do_pool_norm = False")
        update_gen_pool    = False


    model = tensorflow_code.models.xvector(sess, is_test_p, n_spk, tdnn_sizes_before_pooling, tdnn_sizes_after_pooling,
                                           activations_before_pooling, activations_after_pooling,
                                           pool_size=pool_size, it_tr_que=it_tr_que, upd_b_pool=update_b_pool,
                                           upd_a_pool=update_a_pool, upd_multi=update_class, do_feat_norm=do_feat_norm,
                                           upd_feat_norm=update_feat_norm, do_pool_norm=do_pool_norm, upd_pool_norm=update_pool_norm,
                                           use_bug=False, floatX='float32', stop_grad_ap=stop_grad_ap)

    Stats_, embd_A_, _, C_c_ = model( X1_p, C1_p, annoying_train ) # Embeddings + Classifications
        

    # Model that takes embd_A_ from segment 1, and phn features from segment 2 and then predicts features for segment 2.
    tdnn_fe_pred_sizes  = ( [166*7, 166, 7, 1], [166, 166, 1, 1], [166, 166, 1, 1], [166, 166, 1,1], [166,30,1,1] )
    tdnn_fe_activations = [tf.nn.relu] * 4 + [tf.identity]
    #tdnn_fe_activations = [tf.identity] + [tf.nn.leaky_relu] * 3 + [tf.identity]

    
    log.info("Estimating mean and var of bn feats")
    [X, Y, U], _, [[feats, lab], tr_idx], it_batch_phn, it_ctrl_phn = it_tr_phn_que.get_batch()
    P2 = sess.run(tf.one_hot(P2_p, depth=166), {P2_p:lab})          # 
    mean_phn = np.mean(P2, axis=(0,1), keepdims=True)
    var_phn  = np.var(P2, axis=(0,1), keepdims=True)
    log.debug("Mean ofh phn")
    print(mean_phn)
    print(mean_phn.shape)
    log.debug("Var ofh phn")
    print(var_phn)
    print(var_phn.shape)
    
    tdnn_fe_pred = tensorflow_code.models.tdnn_stack(sess, is_test_p, tdnn_sizes=tdnn_fe_pred_sizes, activations=tdnn_fe_activations,
                                                     it_tr_que=[mean_phn, var_phn], upd_tdnn=True, do_feat_norm=False,
                                                     upd_feat_norm=False, side_info_sizes=[512]*5, upd_tdnn_spec=[],
                                                     apply_bn_norm=[True,True,True,True,False], use_bug=False, floatX='float32')

    bnorm_fe_pred  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(sess, mean=np.zeros( 512 ), var=np.ones( 512 ),
                                                                  offset=np.zeros( 512 ),
                                                                  scale=np.ones( 512 ),
                                                                  is_test =is_test_p, decay=0.95,
                                                                  floatX=floatX )

    if stop_reco_grad:
        reco_pred_ = tdnn_fe_pred(2*tf.one_hot(P2_p,depth=166)-1, tf.squeeze(bnorm_fe_pred(tf.expand_dims(tf.stop_gradient(embd_A_), 1))))
    else:
        reco_pred_ = tdnn_fe_pred(2*tf.one_hot(P2_p,depth=166)-1, tf.squeeze(bnorm_fe_pred(tf.expand_dims(embd_A_, 1))))
        
    reco_tgt_ = X2_r_p[:,3:-3,:]    # Need to adjust for context
    #reco_tgt_ = X2_r_p

    # GB model for speaker classification on the embd_A. This will be used to 
    # derive parameters for scoring
    P_GB_  = tf.Variable( np.eye(512).astype('float32'), 'P_')
    mu_GB_ = tf.Variable( np.zeros([512,n_spk]).astype('float32'), 'mu_')

    bnorm_GB  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(sess, mean=np.zeros( 512 ), var=np.ones( 512 ),
                                                             offset=np.zeros( 512 ),
                                                             scale=np.ones( 512 ),
                                                             is_test =is_test_p, decay=0.95,
                                                             floatX=floatX )

    WC_GB_s_ =  tf.matmul(P_GB_,  P_GB_, transpose_b=True  )
    
    embd_A_GB_ = tf.squeeze( bnorm_GB( tf.expand_dims( tf.stop_gradient( embd_A_ ), 1) ) )
    GB_pred_   = tf.tensordot(tf.tensordot( embd_A_GB_, WC_GB_s_, axes=1 ), mu_GB_, axes=1 )
    GB_pred_  -= 0.5 * tf.transpose(tf.reduce_sum(tf.tensordot(WC_GB_s_, mu_GB_, axes=1) * mu_GB_, axis=0) )    

    loss_GB_c_  = tf.tensordot(WGT_c_p, tf.nn.sparse_softmax_cross_entropy_with_logits( labels = L_c_p, logits = GB_pred_ ),
                              axes=[[0], [0]] ) / np.log(n_spk)                            # Normalize with the loss of random predictions 

#############################################################################    
### --- Define loss, train functions etc. ------------------------------- ###
#############################################################################
    mse_weight   = 1e-2
    class_weight = 1.0
    
    loss_c_  = tf.tensordot(WGT_c_p, tf.nn.sparse_softmax_cross_entropy_with_logits( labels = L_c_p, logits = C_c_ ),
                              axes=[[0], [0]] )
    loss_c_  = loss_c_ / np.log(n_spk)                            # Normalize with the loss of random predictions 

    loss_reco_ = tf.losses.mean_squared_error( labels = reco_tgt_, predictions=reco_pred_ )
        
    # Stuff for training "generator",  "speaker classifiers" and feature/bottleneck predictors
    params_to_update_  = model.get_upd_parameters() + tdnn_fe_pred.get_upd_parameters() + bnorm_fe_pred.get_upd_parameters()
    params_to_update_  += [P_GB_] +[ mu_GB_] + bnorm_GB.get_upd_parameters() 

    Loss_1_     = class_weight * loss_c_ + mse_weight * loss_reco_
    Loss_         = Loss_1_ + loss_GB_c_
    grads_, vars_ = list(zip(*optim.compute_gradients(loss = Loss_, var_list = params_to_update_,gate_gradients=optim.GATE_GRAPH))) 
    min_op_       = optim.apply_gradients(list(zip(grads_, vars_)))
    tr_outputs_   = [loss_c_, loss_reco_, Loss_1_, min_op_]  
    tr_outputs_  += [loss_GB_c_]
    
    log.info("Parameters to update:")
    print(params_to_update_)
    def train_function(X1, C1, WGT_c, L_c, X2_r, P2, lr):  
        if not annoying_train:
            tr_inputs_ = {X1_p: X1, C1_p: C1, WGT_c_p: WGT_c, L_c_p: L_c, X2_r_p: X2_r, P2_p: P2, lr_p: lr, is_test_p: False}
        else:
            tr_inputs_ = {X1_p: X1, WGT_c_p: WGT_c, L_c_p: L_c, X2_r_p: X2_r, P2_p: P2, lr_p: lr,
                          is_test_p: False, C1_p: np.array([])}
                        
        return sess.run(tr_outputs_, tr_inputs_ )
    
#############################################################################    
### --- Mini batch generation, Train and dev. functions------------------ ###
#############################################################################    

    # Functions for getting and setting model parameters. During training,
    # these are use to keep the best model parameters, and when we run into
    # the situation that dev. set performance is not improving, we will reset
    # to the best parameters, reduce the learning rate and then try to continue
    # the training.

    # Variable initialization
    sess.run(tf.global_variables_initializer())

    params_to_store_  = model.get_parameters() + tdnn_fe_pred.get_parameters() + bnorm_fe_pred.get_parameters()
    params_to_store_ += [P_GB_] + [mu_GB_] + bnorm_GB.get_parameters() 
    optim_states_     = get_optim_states(optim)
    def get_para():
        return [sess.run(params_to_store_), sess.run(optim_states_)]

    def set_para(para):
        ass_ops = []
        para_   = params_to_store_

        for i in range(len(para_)):
             ass_ops.append( tf.assign( para_[i], para[0][i] ) )

        i = 0
        for v in optim_states_:
            ass_ops.append( tf.assign( v, para[1][i] ) )
            i += 1
        sess.run(ass_ops)
    ### ----     

    
    # To save the model with TF approach. More convenient as architecture is saved to
    # and do not have to be specifed in testing. 
    saver = tf.train.Saver(max_to_keep=100)
    save_func = lambda filename, info: saver.save(sess, filename, global_step=info, write_meta_graph=info==0)

        
    batch_start = 0
    batch_count = batch_start
    
    rng_tr_embd = np.random.RandomState(seed=423+seed_offset)    

    batch_count = 0
    it_tr_embd = utils.mbatch_generation.gen_mbatch_spk_bal(train_scp_info, ivec_dir=None, stats_dir=None, feat_dir=None, stats_order=2,
                                                            frame_step=1, max_length=30000,  
                                                            y_function=None, verbose=False,
                                                            arrange_spk_fcn = None, n_spk_per_batch=batchsize, n_utt_per_spk=n_utt_per_spk,
                                                            output_labs=True, output_scp_ind=False, output_utt_id=True,
                                                            out2put_utt2sideInfo=False, rng=rng_tr_embd )        

    # The batch_iterator class prepares batches in a second thread while the training is running.
    all_bad_utts      = [] # Keeps track of 0-duration utterances. (Which are ignored in trainin). Not used if annoying_train=True  
    it_tr_que_embd    = utils.mbatch_generation.batch_iterator_2(it_tr_embd, load_feats_train_joint,
                                                                 annoying_train, batch_que_length, batch_number=0, use_mpi=False)

    #train_batch_func = utils.train.get_train_batch_multi_reco(it_tr_que_embd, train_function)
    train_batch_func = utils.train.get_train_batch_multi_reco_lab(it_tr_que_embd, train_function, ret_loss=ret_loss) 
        
    # This function will be checked after each epoch. Its output will be used
    # to determine whether to half the learning rate.
    # NOTE: If only one utt, C will be a vector instead of  a matrix!!! This means argmax line will not work!!
    extr_embd_fkn = lambda x, c: sess.run([embd_A_, embd_A_GB_], {X1_p: x, C1_p:c, is_test_p:True})
    def cosine_score(e):
        norm = np.repeat(np.sum(e**2, axis=1, keepdims=True), e.shape[0], 1 )
        norm = np.sqrt((norm*norm.T))
        return np.dot(e, e.T) / norm

    def plda_score(e, m, B, W):
        P, Q, c, k = mBW_2_PQck(m, B, W)
        SP = e.dot(P).dot(e.T)
        SQ = np.repeat(np.sum(e.dot(Q)*e, axis=1, keepdims=True), e.shape[0], 1)
        Sc = np.repeat(e.dot(c)[:,np.newaxis], e.shape[0], 1)
        return SP + SQ + SQ.T + Sc + Sc.T + k

    def check_eer_with_kaldi(scr_mx):
        res = ""
        for i in range(len(dev_trial_enr)):
            enr = dev_trial_enr[i]
            tst = dev_trial_test[i]
            lab = dev_trial_lab[i]
            scr = scr_mx[dev_trial_id2ind[enr], dev_trial_id2ind[tst]] 
            res += str(scr) + " " + lab + "\n"
        command = "compute-eer - 2>/dev/null"
        try:
            eer = subprocess.check_output(command, input=res.encode(), shell=True)
        except subprocess.CalledProcessError:
            log.info("Failed to check EER")        
        return eer.decode()

    def check_dev_class_loss_acc():
        
        X1, C1, X2_r, P2 = load_feats_dev_joint( list(range(len(dev_scp_info['utt2file'])))  )
        
        L_c      = dev_scp_info['utt2spk']
        WGT_c    = np.ones(L_c.shape)/ float(L_c.shape[0])
        loss_c, C, loss_reco, Loss_1, loss_GB_c, Loss = sess.run( [loss_c_, C_c_, loss_reco_, Loss_1_, loss_GB_c_, Loss_],
                                                                  {X1_p: X1, C1_p:C1, WGT_c_p:WGT_c, L_c_p:L_c, X2_r_p:X2_r, P2_p: P2, is_test_p: True} )

        P = np.argmax(C, axis=1)
        Acc = sum(P == L_c)/float(len(L_c))
        log.info("Accuracy: %f, loss_c %f, loss_reco %f, loss_c+loss_reco %f  loss_GB_c %f, Loss %f", Acc, loss_c, loss_reco, Loss_1, loss_GB_c, Loss )
       
        embd_A, embd_A_GB  = extract_embeddings(load_feats_dev_trial, extr_embd_fkn, 512, dev_trial_files, info_offset=0, b_size=2000, n_embds=2)[0]
        scr_mx = cosine_score(embd_A)
        EER = check_eer_with_kaldi(scr_mx)
        log.info("embd_A Cosine,             EER: %s", EER)            

        scr_mx = cosine_score(embd_A_GB)
        EER = check_eer_with_kaldi(scr_mx)
        log.info("embd_A_GB + b-norm - Cosine,          EER: %s", EER)            

        try:        
            mu_GB, P_GB = sess.run([mu_GB_, P_GB_])
            WP = P_GB.dot(P_GB.T)
            E_WP, M_WP = spl.eigh(WP)
            log.info("WP has %d eigenvalues smaller than 1e-6. These will be set to 1e-6" % sum(E_WP<1e-6))
            E_WP = np.maximum(E_WP, 1e-6)
            WP = M_WP.dot(np.diag(E_WP)).dot(M_WP.T)  # Now we should have no problem with singularity
            WP = 0.5*(WP +WP.T)
            WC = np.linalg.inv( WP )             
            BC = np.cov(mu_GB)                        # Between-class covariance (Will be 0 at init so eps is needed. )
            m  = np.mean(mu_GB, axis=1)                                  # dim 512  

            lda_dim  = 150
            D, lda   = spl.eigh( BC, WC )
            lda = lda[:,np.argsort(D)[::-1]][:,:lda_dim]
            embd_A_GB_lda = (embd_A_GB - m).dot(lda) 

            scr_mx = cosine_score(embd_A_GB_lda)
            EER = check_eer_with_kaldi(scr_mx)
            log.info("embd_A + b_norm + LDA + Cosine       EER: %s", EER )

            scr_mx = plda_score(embd_A_GB, m , BC, WC)
            EER = check_eer_with_kaldi(scr_mx)
            log.info("embd_A + b_norm + PLDA,              EER: %s", EER)            

            m = m.dot(lda)
            BC = lda.T.dot(BC).dot(lda) 
            WC = lda.T.dot(WC).dot(lda) 
            scr_mx = plda_score(embd_A_GB_lda, np.zeros(150) , BC, WC)  # Note mean is 0 because we have subtracted it above
            EER = check_eer_with_kaldi(scr_mx)
            log.info("embd_A + b_norm + LDA + PLDA,        EER: %s", EER)            

        except Exception as e:
            log.warning("Failed to estimate trial result with GB")
            log.warning(e.__doc__)
            log.warning(e.message)


        if spk_class_train_only:
            return loss_c
        elif reco_train_only:
            return loss_reco
        else:
            return Loss_1   
        
    ###
    if ( kaldi_txt_model != None ):
        log.info("Loading model %s" % kaldi_txt_model )
        mdl = load_davids_kaldi_model_2(kaldi_txt_model)
        para = load_kaldi_xvec_para_est_stat(mdl, Stats_, it_tr_que, sess, set_para, X1_p, C1_p, is_test_p,
                                             feat_dim=23, n_lay_before_pooling=5, n_lay_after_pooling=2,
                                             feat_norm=do_feat_norm, pool_norm=do_pool_norm)            
        set_para(para)
    elif (tf_model != None):
        para = load_model(tf_model)
        set_para(para)

    
    # This function will call "train_batch" defined above starting with lr_first. After each epoch,
    # check_dev is called, if it is better than before we just call train_batch again. If not, we
    # half the learning rate, reset the parameters to the previous ones and then call train_batch. 
    # This is repeated until we reach lr_last.
    
    # Variable initialization
    if ( kaldi_txt_model == None ) and ( tf_model == None ):
        sess.run( tf.global_variables_initializer() )
    
    start_time  = time.time()
    start_clock = time.clock() 
    log.info( "Starting training at: " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(start_time)) )
    train_nn(n_epoch, n_batch, lr_first, lr_last, train_batch_func, check_dev =check_dev_class_loss_acc, 
             get_para_func =get_para, set_para_func =set_para, model_save_file =model_prefix, patience=patience, save_func=save_func, patience_2=patience_2, half_every_N_epochs = half_every_N_epochs, save_every_epoch=False)
 
    end_time  = time.time()
    end_clock = time.clock() 
    log.info( " Started training at: " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(start_time)) )
    log.info( " Ended training at: " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(end_time)) )
    log.info( " CPU time used: " + str( end_clock - start_clock ) + " s." )

    if (  len(all_bad_utts) > 0 ):
            print("Got a one or more zero-length utterances. This should not happen")
            print("These utterances were discarded but this means batch arrangments") 
            print("for the corresponding speakers might have been suboptimal.") 
            print("SHOULD BE FIXED")
            print(" Utterance(s): ", end=' ')
            print(all_bad_utts)



"""
            para = utils.model_utils.load_model("/mnt/matylda6/rohdin/expts/runs/x-vec_python_train/reconstruction_sitw/exp_10/output//model_epoch-76_lr-1.0_lossTr-0.8917103423178196_lossDev-1.0495161.h5")
            set_para(para)
"""

"""
            mu_GB, P_GB = sess.run([mu_GB_, P_GB_])
            WC = np.linalg.inv( (P_GB.dot(P_GB.T) ) + np.eye(512)*1e-8 ) # Within-class covariance. Need to symmtrize as in training. Also add eps in case it is singular.
            m  = np.mean(mu_GB, axis=1)                                  # dim 512  
            BC = np.cov(mu_GB) + np.eye(512)*1e-8                        # Between-class covariance (Will be 0 at init so eps is needed. )

            lda_dim  = 150
            D, lda   = spl.eigh( BC, WC )
            lda = lda[:,np.argsort(D)[::-1]][:,:lda_dim]
            #lda_full = lda[:,np.argsort(D)[::-1]]
            #lda = lda[:,:lda_dim]
            embd_A_GB_lda = (embd_A_GB - m).dot(lda) 


            D2, lda2   = spl.eigh( P_GB.dot(P_GB.T).dot(np.cov(mu_GB)) )
            #lda = lda2[:,np.argsort(D2)[::-1]][:,:lda_dim]
            lda = lda2[:,np.argsort(D2)[:,:lda_dim]
            embd_A_GB_lda = (embd_A_GB - m).dot(lda) 

            
            scr_mx = cosine_score(embd_A_GB_lda)
            scr = pytel.scoring.Scores(dev_trial_scp_info['spk_name'][ dev_trial_scp_info['utt2spk'] ], dev_trial_scp_info['utt2file'], scr_mx)
            r = compute_results_sre16( [ scr ], [keys[ 'sitw_dev_core-core' ]] )
            log.info("embd_A + b_norm + LDA + Cosine       EER: %f, minDCF001 %f, minDCF0001 %f", r['EER'], r['minDCF001'], r['minDCF0001'])

            #dplda.set_parameter_values( mBW_2_PQck(m[:,np.newaxis], BC, WC) )
            #scr_mx = sess.run(dplda(embd_A_GB_p,embd_A_GB_p), {embd_A_GB_p:embd_A_GB[:,np.newaxis,:]}).squeeze()
            scr_mx = plda_score(embd_A_GB, m , BC, WC)
            print scr_mx.shape
            scr = pytel.scoring.Scores(dev_trial_scp_info['spk_name'][ dev_trial_scp_info['utt2spk'] ], dev_trial_scp_info['utt2file'], scr_mx)
            r = compute_results_sre16( [ scr ], [keys[ 'sitw_dev_core-core' ]] )
            log.info("embd_A + b_norm + PLDA               EER: %f, minDCF001 %f, minDCF0001 %f", r['EER'], r['minDCF001'], r['minDCF0001'])

            m = m.dot(lda)
            #BC = lda_full.T.dot(BC).dot(lda_full)[:lda_dim,:lda_dim]
            BC = lda.T.dot(BC).dot(lda) 
            WC = lda.T.dot(WC).dot(lda) 
            #dplda_lda.set_parameter_values( mBW_2_PQck(m[:,np.newaxis], BC, WC) )
            #scr_mx = sess.run(dplda_lda(embd_A_GB_p,embd_A_GB_p), {embd_A_GB_p:embd_A_GB.dot(lda)[:,np.newaxis,:]}).squeeze()
            plda_score(embd_A_GB_lda, np.zeros(150) , WC, BC)  # Note mean is 0 because we have subtracted it above
            print scr_mx.shape
            scr = pytel.scoring.Scores(dev_trial_scp_info['spk_name'][ dev_trial_scp_info['utt2spk'] ], dev_trial_scp_info['utt2file'], scr_mx)
            r = compute_results_sre16( [ scr ], [keys[ 'sitw_dev_core-core' ]] )
            log.info("embd_A + b_norm + LDA + PLDA         EER: %f, minDCF001 %f, minDCF0001 %f", r['EER'], r['minDCF001'], r['minDCF0001'])
"""
"""
            min_n_avl_samp = np.min( n_avl_samp )
            max_len        = np.min( [min_n_avl_samp+1, max_length] ) # Need to add 1 because max_len because the intervall is [min_len, max_len)?????!!!!???
            n_sel_samp     = rng_f_tr.randint(min_length, max_len)       # not [min_len, max_len]

            start = []
            end = []
            for i,f in enumerate(files):
                # The start_from_zero option is mainly for debugging/development
                if start_from_zero:
                    start.append(0)
                else:
                    last_possible_start = n_avl_samp[i] - n_sel_samp
                    start.append(rng_f_tr.randint(0,  last_possible_start + 1)[0] )# This means the intervall is [0,last_possible_start + 1) = [0, last_possible_start]
                    end.append(start[-1] + n_sel_samp)


            data = kaldi_io.read_file_segm(files, start, end)
            data = np.stack( data, axis=0 )
            print(data.shape)
            return data
            """
"""
            start_from_zero = False
            n_avl_samp     = np.array( [train_scp_info['utt2sideInfo'][uu] for uu in u] ) 
            files          = [train_scp_info['utt2file'][uu] for uu in u] 
            min_n_avl_samp = np.min( n_avl_samp )
            max_len        = np.min( [min_n_avl_samp+1, max_length] ) # Need to add 1 because max_len because the intervall is [min_len, max_len)?????!!!!???


            #if ( segm_B_clean >=1 ):
            #    lab = [ np.genfromtxt(lab_dir + fix_names(f) + ".hp") for f in files  ]
            #else:
            lab = [ np.genfromtxt(lab_dir + f + ".hp") for f in files  ]

            
            n_sel_samp_A     = rng_f_tr.randint(min_length, max_len)       # not [min_len, max_len]
            n_sel_samp_B     = rng_f_tr.randint(min_length, max_len)       # not [min_len, max_len]
            
            start_A = []
            start_B = []
            end_A = []
            end_B = []
            for i,f in enumerate(files):
                # The start_from_zero option is mainly for debugging/development
                if start_from_zero:
                    start_A.append(0)
                    start_B.append(0)
                else:
                    last_possible_start_A = n_avl_samp[i] - n_sel_samp_A
                    start_A.append(rng_f_tr.randint(0,  last_possible_start_A + 1)[0] )# This means the intervall is [0,last_possible_start + 1) = [0, last_possible_start]
                    end_A.append(start_A[-1] + n_sel_samp_A)
                    last_possible_start_B = n_avl_samp[i] - n_sel_samp_B
                    start_B.append(rng_f_tr.randint(0,  last_possible_start_B + 1)[0] )# This means the intervall is [0,last_possible_start + 1) = [0, last_possible_start]
                    end_B.append(start_B[-1] + n_sel_samp_B)

            data_A = kaldi_io.read_file_segm(files, start_A, end_A)
            data_B = kaldi_io.read_file_segm(files, start_B, end_B)
            data_A = np.stack( data_A, axis=0 )
            data_B = np.stack( data_B, axis=0 )
            lab[i]=lab[i][start_B:end_B]
            print(data_A.shape)
            print(data_B.shape)
            return [data_A, data_B, lab]
"""
