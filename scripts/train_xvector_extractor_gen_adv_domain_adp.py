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


floatX='float32'


import sys, os, cPickle, copy, subprocess, time, inspect #, re, h5py 

from utils.misc import get_logger
log = get_logger()

    
from tensorflow_code import pool_fkns

import numpy as np
import utils.mbatch_generation
from utils.load_data import *
import tensorflow_code.optim
import tensorflow_code.models
import tensorflow as tf
from utils.train import train_nn, get_train_batch_fkn  
from tensorflow_code.dplda import mBW_2_PQck
from tensorflow_code.dplda import p_eff, llrThreshold, labMat2weight, lab2matrix
from tensorflow_code.load_save import save_tf_model

from tensorflow_code.nn_def import tf_ff_nn

from utils.misc import make_dir_mpi

#from utils.model_utils import load_model

if ( __name__ == "__main__" ):
    
    PRISM = True
    
    # Check hostname and cpu info. Will be printed in log below. Cpu info just checks
    # the first cpu on the machine, not necessarily the one we use but normally they
    # are the same.
    host_name  = os.uname()[1]
    cpu_info   = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | head -n 1 ", shell=True).split(':')[1]

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
        
    if (not PRISM):
        #feats_dir         = '/mnt/matylda6/rohdin/expts/runs/feat_2_score_nn_tf_pytel/test_1/feats_proc/'
        feats_dir         = '/tmp/rohdin/feats_proc/'
    else:
        #feats_dir          = '/mnt/matylda6/rohdin/expts/runs/x-vec_PRISM_base_expts/data_prep/feats_eval_test/'
        feats_dir         = '/mnt/ssd/rohdin/feats/'

    # We will check whether data exists in the below directories.
    # If this data doesn't exisit, we will not use the file.
    # The reason for checking e.g. i-vectors even though we don't
    # use them is that we may want to make sure only the same files
    # as some baseline is used.
    ivec_dir_check    = None
    stats_dir_check   = None 
    feats_dir_check   = feats_dir
   
    # Output model
    model_prefix = output_dir + '/model'
        
### --- Data etc. ------------------------------------------ ###
    if (not PRISM):
        train_scp   = "/mnt/matylda6/rohdin/expts/runs/feat_2_score_nn_tf_pytel/test_1/scp/train_large_info_no_time.scp"
        dev_scp     = "/mnt/matylda6/rohdin/expts/runs/feat_2_score_nn_tf_pytel/test_1/scp/valid_jhu_segm.scp"
        unl_scp =''
    else:
        train_scp   = "/mnt/matylda6/rohdin/expts/runs/x-vec_PRISM_base_expts/data_prep/scp/train_red_w_id_dirfix.scp"
        dev_scp     = "/mnt/matylda6/rohdin/expts/runs/x-vec_PRISM_base_expts/data_prep/scp/valid_jhu_segm_fix.scp"
        unl_scp     = "/mnt/matylda5/iplchot/SRE/NIST2018/lists/v2/SRE18_DEV_UNLABELED.scp"

    
    train_scp_info_file   = output_dir +'train_scp_info.pkl'
    dev_scp_info_file     = output_dir +'dev_scp_info.pkl'
    unl_scp_info_file     = output_dir +'unl_scp_info.pkl'

    
    # File length limits
    min_length        = 200
    max_length        = 400 
    #min_length        = 300
    #max_length        = 301 

    
### --- Training settings ------------------------------- ###

    batchsize_unl  = 150
        
    annoying_train = True

    bn_decay   = 0.95 # Decay of moving average in batchnorm (for estimating stats used in testing).     

    n_utt_per_spk = 1
    batchsize  = 150
    n_epoch    = 5000            # Maximum number of epochs
    n_batch    = 400             # Number of minimbatches per epoch (dev loss is checked after each epoch) 
    lr_first   = 1e-0            # Initial learning rate. Will be halved if there is no improvent on dev loss
    lr_last    = 1e-5            # Last learning rate. When we have reduced the learning rate below this value  
                                 # the training stops. See the training scheme below.

    patience  = 31               # For one learning rate, the training is allowed to fail this many times.
                                 # If it fails one more time, the learning rate will be halved, parameters
                                 # reset and the training continued.
    patience_2 = 3
                                 
    batch_que_length = 2         # Number of batch in que (will be prepared in parallel with training)    
   
    lr_p, optim, get_optim_states = tensorflow_code.optim.get_optimizer('SGD', floatX=floatX)
    log.info("Optimizer: " + optim._name)

#############################################################################            
### --- Load and prepare data ------------------------------------------- ###
#############################################################################    

    # Create functions for loading data given a set of files. 
    if annoying_train:
        # All segments will be of the same length
        rng_f_tr = np.random.RandomState(seed=519)
        def load_feats_train(files):
            #return load_jhu_feat_segm_fixed_len(feats_dir, files, min_length, max_length, floatX='float32', start_from_zero=True, rng=rng_f_tr)
            return load_jhu_feat_segm_fixed_len(feats_dir, files, min_length, max_length, floatX='float32', rng=rng_f_tr)
    else:
        # Segments can be of different lenght
        rng_f_tr = np.random.RandomState(seed=519)
        def load_feats_train(files):
            return load_jhu_feat_segm(feats_dir, files, min_length, max_length, floatX='float32', rng=rng_f_tr)        

    # The full file length is used.    
    def load_feats_dev(files):
        return load_jhu_feat(feats_dir, files, floatX='float32')


    # Prepare train scp info
    if os.path.exists( train_scp_info_file ):

        log.info("Loading scp info from " + train_scp_info_file)
        inp            = open( train_scp_info_file, 'r')
        train_scp_info = cPickle.load(inp)
        inp.close()

    else:
        train_scp_info = utils.mbatch_generation.get_scp_info(train_scp, ivec_dir_check, stats_dir_check, feats_dir_check)
        # We need to map the speaker IDs to the original names in order to have consistency with
        # models trained by Kaldi (in case we want to load them). Also, we need consistency
        # between the training and development set. Luckily, the speaker names are just integers
        # in this data set so doing this is easy.
        if (not PRISM):
            for i in range(len(train_scp_info['utt2spk'])):
                 train_scp_info['utt2spk'][ i ] = int(train_scp_info['spk_name'][train_scp_info['utt2spk'][ i ]])

            sc = copy.deepcopy(train_scp_info['spk_counts'])
            for i in range(len(train_scp_info['spk_counts'])):
                train_scp_info['spk_counts'][ int(train_scp_info['spk_name'][i] )] = sc[i]

        log.info( "Saving scp to " + train_scp_info_file )
        out = open(train_scp_info_file, 'w')
        cPickle.dump(train_scp_info, out)
        out.close()

    # Prepare dev (valid) scp info
    if os.path.exists( dev_scp_info_file ):

        log.info("Loading scp info from " + dev_scp_info_file)
        inp          = open( dev_scp_info_file, 'r')
        dev_scp_info = cPickle.load(inp)
        inp.close()

    else:
        dev_scp_info = utils.mbatch_generation.get_scp_info(dev_scp, ivec_dir_check, stats_dir_check, feats_dir_check)
        # Map the speaker IDs to the original "names"
        if (not PRISM):
            for i in range(len(dev_scp_info['utt2spk'])):
                 dev_scp_info['utt2spk'][ i ] = int(dev_scp_info['spk_name'][dev_scp_info['utt2spk'][ i ]])

            sc = copy.deepcopy(dev_scp_info['spk_counts'])
            dev_scp_info['spk_counts'] = np.zeros( train_scp_info['spk_counts'].shape  )
            for i in range(len(sc)):
                dev_scp_info['spk_counts'][ int(dev_scp_info['spk_name'][i] )] = sc[i]

        # We need to make the dev spk IDs match the ones of the training scp
        # NOTE that this means utt2spk will no correspond to spk_name or spk_counts anymore.
        else:
            train_spkname2spkid = dict(zip(train_scp_info['spk_name'], range(len(train_scp_info['spk_name']))))
            for i in range(len(dev_scp_info['utt2spk'])):

                spk_name = dev_scp_info['spk_name'][ dev_scp_info['utt2spk'][i] ]
                dev_scp_info['utt2spk'][ i ] = train_spkname2spkid[ spk_name ]

        log.info( "Saving scp to " + dev_scp_info_file )
        out = open(dev_scp_info_file, 'w')
        cPickle.dump(dev_scp_info, out)
        out.close()


    n_spk     = train_scp_info['spk_counts'].shape[0]    


    #### Load scp for unlabeled set - WARNING: SCP IS ASSUMED TO HAVE UNIQUE SPEAKER LABELS
    if os.path.exists( unl_scp_info_file ):

        log.info("Loading scp info from " + unl_scp_info_file)
        inp            = open( unl_scp_info_file, 'r')
        unl_scp_info = cPickle.load(inp)
        inp.close()

    else:
        unl_scp_info = utils.mbatch_generation.get_scp_info(unl_scp, ivec_dir_check, stats_dir_check, feats_dir_check)
        log.info( "Saving scp to " + unl_scp_info_file )
        out = open(unl_scp_info_file, 'w')
        cPickle.dump(unl_scp_info, out)
        out.close()

    # Make a sanity check the we have unique speaker IDs
    assert( len(unl_scp_info['utt2spk']) == len(unl_scp_info['spk_name']) )
    ####
    
    
    # Detect which GPU to use
    command='nvidia-smi --query-gpu=memory.free,memory.total --format=csv |tail -n+2| awk \'BEGIN{FS=" "}{if ($1/$3 > 0.98) print NR-1}\''
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = subprocess.check_output(command, shell=True).rsplit('\n')[0]
        log.info("CUDA_VISIBLE_DEVICES " + os.environ["CUDA_VISIBLE_DEVICES"])
    except subprocess.CalledProcessError:
        log.info("No GPU seems to be available")        
    sess            = tf.Session()


    # A generator for the training data. We can't use a global random number
    # in the case when noising and MPI was used together (batches may get out
    # of sync for different workers) so we make one here that will be give to
    # the batch generator.
    rng_tr = np.random.RandomState(seed=123)    

    batch_count = 0
    it_tr=utils.mbatch_generation.gen_mbatch_spk_bal(train_scp_info, ivec_dir=None, stats_dir=None, feat_dir=None, stats_order=2,
                                                     frame_step=1, max_length=30000,  
                                                     y_function=None, verbose=False,
                                                     arrange_spk_fcn = None, n_spk_per_batch=batchsize, n_utt_per_spk=n_utt_per_spk,
                                                     output_labs=True, output_scp_ind=False, output_utt_id=True,
                                                     rng=rng_tr )        

    # The batch_iterator class prepares batches in a second thread while the training is running.
    all_bad_utts = [] # Keeps track of 0-duration utterances. (Which are ignored in trainin). Not used if annoying_train=True  
    it_tr_que    = utils.mbatch_generation.batch_iterator(it_tr, train_scp_info, load_feats_train,
                                                          annoying_train, batch_que_length, batch_number=0, use_mpi=False)

   

    
#############################################################################    
### --- Set up the model ------------------------------------------------ ###
#############################################################################

    # Variables. 
    X1_p        = tf.placeholder(floatX, shape=[None,None,23], name='X1_p') # Features 
    C1_p        = tf.placeholder(dtype='int32', shape=[None], name='C1_p')  # Indices
    M1_p        = tf.placeholder(floatX, shape=[None,512], name='M1_p')     # Input to critic (Used in critic training)
    h1_p         = tf.placeholder(floatX, shape=[None,512], name='h1_p')    # Averaged input to critic (Used in critic training)
    is_test_p   = tf.placeholder(dtype='bool', shape=[], name='is_test_p')  # Tells whether it is the training or the testing phase.
    WGT_c_p     = tf.placeholder(floatX, shape=[None], name='WGT_c_p')      # Weight matrix for class
    L_c_p       = tf.placeholder('int32', shape=[None], name='L_c_p')       # Label vector for class
                                                                            
    # Model This is the default in fact.
    tdnn_sizes_before_pooling   = ( [115, 512, 5, 1], [1536, 512, 3, 2], [1536,512, 3, 3], [512, 512, 1,1], [512,1500,1,1] )
    tdnn_sizes_after_pooling    = ( [3000, 512, 1, 1], [512, 512, 1, 1] )
    activations_before_pooling  = [tf.nn.leaky_relu] * 5
    activations_after_pooling   = [tf.nn.leaky_relu] * 2
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
                                           pool_size=pool_size, it_tr_que=it_tr_que, 
                                           upd_b_pool=update_b_pool,
                                           upd_a_pool=update_a_pool, upd_multi=update_class,
                                           do_feat_norm=do_feat_norm, upd_feat_norm=update_feat_norm,
                                           do_pool_norm=do_pool_norm, upd_pool_norm=update_a_pool, floatX='float32')

    _, embd_A_, _, C_c_ = model( X1_p, C1_p, annoying_train ) # Embeddings + Classifications      

    # Set up the critic network
    critic_sizes = [512, 256, 256, 1 ]
    critic_para  = tensorflow_code.initializers.init_params_simple_he_uniform( critic_sizes, floatX=floatX)     
    critic_nn    = tf_ff_nn(sess, critic_para, act_fkns =[tf.nn.leaky_relu] * 2, name='critic_nn', floatX=floatX)

    log.info("Summary of critic network")
    critic_nn.summary()
    
    critic_out_   = critic_nn(embd_A_)
    
#############################################################################    
### --- Define loss, train functions etc. ------------------------------- ###
#############################################################################    
    gamma    = 5       # 10.0
    delta    = 0.1     # 1.0
    alpha_1  = 0.001   # 0.01
    alpha_2  = 1.0
    n_w_iter = 5      # 5
    
    n_in     = batchsize_unl
    n_out    = batchsize * n_utt_per_spk

    loss_w_ =  1.0 / n_out * tf.reduce_sum( critic_out_[0:n_out]  ) - 1.0 / n_in * tf.reduce_sum( critic_out_[n_out : n_out + n_in] ) 
                                                         
    loss_grad_ = tf.gradients(critic_nn(h1_p), h1_p) #[0] ?!!!!!!!!
    loss_grad_ = (tf.norm(loss_grad_) -1 ) **2       # !!!!

    ### We only take the first n_out predictions to the classification loss
    loss_c_  = tf.tensordot(WGT_c_p, tf.nn.sparse_softmax_cross_entropy_with_logits( labels = L_c_p, logits = C_c_[0:n_out] ),
                            axes=[[0], [0]] )
    loss_c_  = loss_c_ / np.log(n_spk)                            # Normalize with the loss of random predictions 
    loss_c_ += model.get_l2_reg(l2_reg_b_pool, l2_reg_a_pool, l2_reg_class)

    ### Need to do anothor one for dev set since we don't split here
    loss_c_dev_  = tf.tensordot(WGT_c_p, tf.nn.sparse_softmax_cross_entropy_with_logits( labels = L_c_p, logits = C_c_ ), axes=[[0], [0]] )
    loss_c_dev_  = loss_c_dev_ / np.log(n_spk)                    # Normalize with the loss of random predictions 
    loss_c_dev_ += model.get_l2_reg(l2_reg_b_pool, l2_reg_a_pool, l2_reg_class)

    
    # Stuff for critic (will be done several times per training iteration)
    Loss_w_grad_ = -(loss_w_ - gamma * loss_grad_)                 # We take the negative of this since the critic wants to maxize it.
    log.warning("WARNING: Last bias in critic is not updated.")# It's current value is %f", critic_nn.get_parameter_values()[-1])
    params_to_update_critic  = critic_nn.get_upd_parameters()[:-1] # We excluded the last bias (scalar). It is intialized to 0 so this way it is effectively not used.
    log.info("Parameters to update in critic inner training loop:")
    print params_to_update_critic                                            
    grads_w_, vars_w_     = zip(*optim.compute_gradients(loss = Loss_w_grad_, var_list = params_to_update_critic,
                                                         gate_gradients=optim.GATE_GRAPH)) 

    def get_w_input(X1, C1):
        inputs_ = {X1_p: X1, C1_p: np.array([]), is_test_p: False}
        outputs_ = embd_A_
        return sess.run(outputs_, inputs_ )

    
    min_op_w     = optim.apply_gradients(zip(grads_w_, vars_w_))                 ### M1 needed ??
    tr_outputs_w_   = [loss_w_, loss_grad_, Loss_w_grad_, min_op_w]  
    def train_function_w(M1, h1, lr):   
        tr_inputs_w_ = {embd_A_: M1, h1_p: h1, lr_p: lr, is_test_p: False}
        return sess.run(tr_outputs_w_, tr_inputs_w_ )

    
    # Stuff for training "generator" and speaker classifier network
    params_to_update_g_c_  = model.get_upd_parameters()
    Loss_w_c_ = loss_c_ + delta * loss_w_

    grads_g_c_, vars_g_c_     = zip(*optim.compute_gradients(loss = Loss_w_c_, var_list = params_to_update_g_c_,
                                                         gate_gradients=optim.GATE_GRAPH)) 
    min_op_g_c_    = optim.apply_gradients(zip(grads_g_c_, vars_g_c_))
    tr_outputs_g_c_   = [loss_w_, loss_c_, Loss_w_c_, min_op_g_c_]  

    log.info("Parameters to update in generator and classifier:")
    print params_to_update_g_c_
    def train_function_g_c(X1, C1, WGT_c, L_c, lr):
        if not annoying_train:
            tr_inputs_g_c_ = {X1_p: X1, C1_p: C1, WGT_c_p: WGT_c, L_c_p: L_c, lr_p: lr, is_test_p: False}
        else:
            tr_inputs_g_c_ = {X1_p: X1, WGT_c_p: WGT_c, L_c_p: L_c, lr_p: lr, is_test_p: False, C1_p: np.array([])}
                        
        return sess.run(tr_outputs_g_c_, tr_inputs_g_c_ )
                    
    """
    model_file = "/mnt/matylda6/rohdin/expts/runs/x-vec_python_train/test_167/output/model-100"
    log.info("Restoring model %s", model)
    saver = tf.train.import_meta_graph(re.sub('-\d+','-0.meta', model_file))
    graph = tf.get_default_graph()
    #print re.sub('/model\-\d+','', model)
    #saver.restore(sess, "./model-17")
    saver.restore(sess, os.path.relpath(model_file)) 
    """
    
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



    
    ### --- Remove this way of saving when the below is tested properly
    params_to_store_ = model.get_parameters() + critic_nn.get_parameters()
    optim_states_    = get_optim_states(optim)
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
    #para=load_model('/mnt/matylda6/rohdin/expts/runs/x-vec_python_train/test_167/output/model_epoch-100_lr-1.0_lossTr-0.167048340328_lossDev-0.22647.h5')
    #set_para(para)
        

    
    # To save the model with TF approach. More convenient as architecture is saved to
    # and do not have to be specifed in testing. 
    saver = tf.train.Saver(max_to_keep=100)
    save_func = lambda filename, info: saver.save(sess, filename, global_step=info, write_meta_graph=info==0)

        
    batch_start = 0
    batch_count = batch_start


    ####
    # The batch_iterator class prepares batches in a second thread while the training is running.
    all_bad_utts = [] # Keeps track of 0-duration utterances. (Which are ignored in trainin). Not used if annoying_train=True  

    rng_tr = np.random.RandomState(seed=231)    

    batch_count = 0
    it_tr=utils.mbatch_generation.gen_mbatch_spk_bal(train_scp_info, ivec_dir=None, stats_dir=None, feat_dir=None, stats_order=2,
                                                     frame_step=1, max_length=30000,  
                                                     y_function=None, verbose=False,
                                                     arrange_spk_fcn = None, n_spk_per_batch=batchsize, n_utt_per_spk=n_utt_per_spk,
                                                     output_labs=True, output_scp_ind=False, output_utt_id=True,
                                                     rng=rng_tr )        


    # Data generator for unlabeled data set
    rng_unl = np.random.RandomState(seed=673)    
    it_unl  = utils.mbatch_generation.gen_mbatch_spk_bal(unl_scp_info, ivec_dir=None, stats_dir=None, feat_dir=None, stats_order=2,
                                                         frame_step=1, max_length=30000,  
                                                         y_function=None, verbose=False,
                                                         arrange_spk_fcn = None, n_spk_per_batch=batchsize_unl, n_utt_per_spk=1,
                                                         output_labs=True, output_scp_ind=False, output_utt_id=True,
                                                         rng=rng_unl )        
    # Combines the two iterators
    it_que = utils.mbatch_generation.batch_iterator_multi_set([it_tr, it_unl], [train_scp_info, unl_scp_info],
                                                              load_feats_train, annoying_train, batch_que_length,
                                                              batch_number=0, use_mpi=False)
    #####
    
    # Function that will do training on one batch
    rng_w = np.random.RandomState(seed=501)    
    train_batch_func = utils.train.get_train_w_g_c_batch_fcn( it_que, get_w_input, train_function_w,
                                                              n_w_iter, alpha_1, rng_w, train_function_g_c, alpha_2 )

    
    # This function will be checked after each epoch. Its output will be used
    # to determine whether to half the learning rate.
    # NOTE: If only one utt, C will be a vector instead of  a matrix!!! This means argmax line will not work!!
    def check_dev_class_loss_acc():
        X1, C1 = load_feats_dev( dev_scp_info['utt2file']  )
        L_c    = dev_scp_info['utt2spk']
        WGT_c = np.ones(L_c.shape)/ float(L_c.shape[0])
        L, C = sess.run([loss_c_dev_, C_c_], {X1_p: X1, C1_p:C1, WGT_c_p:WGT_c, L_c_p:L_c, is_test_p: True} )
        P = np.argmax(C, axis=1)
        Acc = sum(P == L_c)/float(len(L_c))
        log.info("Loss %f, Accuracy: %f", L, Acc )
        return L   
    
    # This function will call "train_batch" defined above starting with lr_first. After each epoch,
    # check_dev is called, if it is better than before we just call train_batch again. If not, we
    # half the learning rate, reset the parameters to the previous ones and then call train_batch. 
    # This is repeated until we reach lr_last.
    
    # Variable initialization
    sess.run( tf.global_variables_initializer() )
    
    start_time  = time.time()
    start_clock = time.clock() 
    log.info( "Starting training at: " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(start_time)) )
    train_nn(n_epoch, n_batch, lr_first, lr_last, train_batch_func, check_dev =check_dev_class_loss_acc, 
             get_para_func =get_para, set_para_func =set_para, model_save_file =model_prefix, patience=patience, save_func=save_func, patience_2=patience_2)
 
    end_time  = time.time()
    end_clock = time.clock() 
    log.info( " Started training at: " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(start_time)) )
    log.info( " Ended training at: " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(end_time)) )
    log.info( " CPU time used: " + str( end_clock - start_clock ) + " s." )

    if (  len(all_bad_utts) > 0 ):
            print "Got a one or more zero-length utterances. This should not happen"
            print "These utterances were discarded but this means batch arrangments" 
            print "for the corresponding speakers might have been suboptimal." 
            print "SHOULD BE FIXED"
            print " Utterance(s): ",
            print all_bad_utts




