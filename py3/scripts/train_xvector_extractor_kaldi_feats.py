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


import sys, os, pickle, copy, subprocess, time, inspect 

from utils.misc import get_logger
log = get_logger()

    
from tensorflow_code import pool_fkns

import numpy as np
import utils.mbatch_generation
from utils.load_data import *
import tensorflow_code.optim
import tensorflow_code.models
import tensorflow as tf
from utils.train import train_nn, get_train_batch_multi_fkn  
from tensorflow_code.dplda import mBW_2_PQck
from tensorflow_code.dplda import dplda_simple
from tensorflow_code.load_save import save_tf_model
#from kaldi_io import get_durations_file_list
import kaldi_io
from utils.model_utils import load_davids_kaldi_model_2
from utils.model_utils import load_model

from utils.misc import make_dir_mpi

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

    train_scp   = "/workspace/jrohdin/expts/tests/tf_xvector/feats_train_w_spk_durations.scp"
    dev_scp     = "/workspace/jrohdin/expts/tests/tf_xvector/feats_valid_w_spk_seg_times.scp"

    train_scp_info_file   = output_dir +'train_scp_info.pkl'
    dev_scp_info_file     = output_dir +'dev_scp_info.pkl'
      
    
    # File length limits
    min_length        = 200
    max_length        = 400 
    #min_length        = 300
    #max_length        = 301 

    
### --- Training settings ------------------------------- ###
   
        
    annoying_train = True

    bn_decay   = 0.95 # Decay of moving average in batchnorm (for estimating stats used in testing).     

    n_utt_per_spk = 2               # Number of utterances per speaker in the batch
    batchsize     = 150             # Number of speakers in the batch
    n_epoch       = 5000            # Maximum number of epochs
    n_batch       = 400             # Number of minimbatches per epoch (dev loss is checked after each epoch) 
    lr_first      = 1e-0            # Initial learning rate. Will be halved if there is no improvent on dev loss
    lr_last       = 1e-5            # Last learning rate. When we have reduced the learning rate below this value  
                                 # the training stops. See the training scheme below.

    patience  = 31               # For one learning rate, the training is allowed to fail this many times.
                                 # If it fails one more time, the learning rate will be halved, parameters
                                 # reset and the training continued.
    patience_2 = -1              # Patience 2 will be used after 150 epochs if != -1
                                 
    batch_que_length = 2         # Number of batch in que (will be prepared in parallel with training)    
   
    lr_p, optim, get_optim_states = tensorflow_code.optim.get_optimizer('SGD', floatX=floatX)
    log.info("Optimizer: " + optim._name)

    kaldi_txt_model = None #'/workspace/jrohdin/expts/tests/tf_xvector/final.v2.txt'
    tf_model        = "/workspace/jrohdin/expts/tf_xvector_baselines/exp_1/output/model_epoch-28_lr-1.0_lossTr-0.3992812346667051_lossDev-0.4422987.h5"
    lr_first        = 1.0
    seed_offset     = 28 * 1357 
    assert( (kaldi_txt_model == None) or (tf_model == None) )
    
    
#############################################################################            
### --- Load and prepare data ------------------------------------------- ###
#############################################################################    

    # Create functions for loading data given a set of files. 
    if annoying_train:
        # All segments will be of the same length
        rng_f_tr = np.random.RandomState( seed = 519 + seed_offset )
        def load_feats_train(u):
            start_from_zero = False
            #n_avl_samp     = np.array( [train_scp_info['durations'][uu] for uu in u] )
            n_avl_samp     = np.array( [train_scp_info['utt2sideInfo'][uu] for uu in u] ) 
            files          = [train_scp_info['utt2file'][uu] for uu in u] 
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

            #print(files)
            #print(start)
            #print(end)
            data = kaldi_io.read_file_segm(files, start, end)
            data = np.stack( data, axis=0 )
            print(data.shape)
            return data
                
            
    else:
        # Segments can be of different lenght
        rng_f_tr = np.random.RandomState( seed = 519 + seed_offset )
        raise ("Not implemented" )

    # The full file length is used.    
    def load_feats_dev(u):
        #print (u)
        #print(dev_scp_info['utt2sideInfo'])
        start = [dev_scp_info['utt2sideInfo'][uu][0] for uu in u]
        end   = [dev_scp_info['utt2sideInfo'][uu][1] for uu in u]
        files = [dev_scp_info['utt2file'][uu] for uu in u] 
        data = kaldi_io.read_file_segm(files, start, end)
        durations = np.array( end ) - np.array( start )
        idx   = np.cumsum( durations )
        idx   = np.insert(idx, 0,0)
        return np.vstack( data )[np.newaxis,:,:], idx


    # Prepare train scp info
    if os.path.exists( train_scp_info_file ):

        log.info("Found scp info: " + train_scp_info_file)
        inp            = open( train_scp_info_file, 'rb')
        train_scp_info = pickle.load(inp)
        log.info("Loaded scp info with " + str(len(train_scp_info['utt2spk'])) +  " utterances and " + str(len(train_scp_info['spk_name'])) + " speakers.")
        inp.close()

    else:
        train_scp_info = utils.mbatch_generation.get_kaldi_scp_info( train_scp )
        #train_scp_info['durations'] = get_durations_file_list( train_scp_info['utt2file'] )
        #print (train_scp_info['durations'])
        #sys.exit(-1)
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
    
    # Detect which GPU to use
    command='nvidia-smi --query-gpu=memory.free,memory.total --format=csv |tail -n+2| awk \'BEGIN{FS=" "}{if ($1/$3 > 0.98) print NR-1}\''
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = subprocess.check_output(command, shell=True).decode('utf-8').rsplit('\n')[0]
        log.info("CUDA_VISIBLE_DEVICES " + os.environ["CUDA_VISIBLE_DEVICES"])
    except subprocess.CalledProcessError:
        log.info("No GPU seems to be available")        
    sess            = tf.Session()


    # A generator for the training data. We can't use a global random number
    # in the case when noising and MPI was used together (batches may get out
    # of sync for different workers) so we make one here that will be give to
    # the batch generator.
    rng_tr = np.random.RandomState( seed = 123 + seed_offset )    

    batch_count = 0
    it_tr=utils.mbatch_generation.gen_mbatch_spk_bal(train_scp_info, ivec_dir=None, stats_dir=None, feat_dir=None, stats_order=2,
                                                     frame_step=1, max_length=30000,  
                                                     y_function=None, verbose=False,
                                                     arrange_spk_fcn = None, n_spk_per_batch=batchsize, n_utt_per_spk=n_utt_per_spk,
                                                     output_labs=True, output_scp_ind=False, output_utt_id=True,
                                                     rng=rng_tr )        

    # The batch_iterator class prepares batches in a second thread while the training is running.
    all_bad_utts = [] # Keeps track of 0-duration utterances. (Which are ignored in trainin). Not used if annoying_train=True  
    it_tr_que    = utils.mbatch_generation.batch_iterator_2(it_tr, load_feats_train,
                                                            annoying_train, batch_que_length, batch_number=0, use_mpi=False)
   
    
#############################################################################    
### --- Set up the model ------------------------------------------------ ###
#############################################################################

    # Variables. 
    X1_p        = tf.placeholder(floatX, shape=[None,None,23], name='X1_p') # Features 
    C1_p        = tf.placeholder(dtype='int32', shape=[None], name='C1_p')  # Indices
    is_test_p   = tf.placeholder(dtype='bool', shape=[], name='is_test_p')  # Tells whether it is the training or the testing phase.
    WGT_m_p     = tf.placeholder(floatX, shape=[None], name='WGT_m_p')      # Weight matrix for multi
    L_m_p       = tf.placeholder('int32', shape=[None], name='L_m_p')       # Label vector for multi
                                                                            
    # Model This is the default in fact.
    tdnn_sizes_before_pooling   = ( [115, 512, 5, 1], [1536, 512, 3, 2], [1536,512, 3, 3], [512, 512, 1,1], [512,1500,1,1] )
    tdnn_sizes_after_pooling    = ( [3000, 512, 1, 1], [512, 512, 1, 1] )
    activations_before_pooling  = [tf.nn.relu] * 5
    activations_after_pooling   = [tf.nn.relu] * 2
    pool_size                   = 3000

    do_feat_norm   = False
    do_pool_norm   = True

    # Which part of the network to update
    update_feat_norm   = False
    update_b_pool      = True
    update_pool_norm   = False
    update_a_pool      = True
    update_multi       = True

    # Regularization. Used a bit further below in the code.
    l2_reg_b_pool = 0.0 
    l2_reg_a_pool = 0.0 
    l2_reg_multi  = 0.0 
    
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
                                           upd_a_pool=update_a_pool, upd_multi=update_multi,
                                           do_feat_norm=do_feat_norm, upd_feat_norm=update_feat_norm,
                                           do_pool_norm=do_pool_norm, upd_pool_norm=update_pool_norm, floatX='float32')

    Stats_, _, _, C_m_ = model( X1_p, C1_p, annoying_train ) # Classifications      

    log.info("Resetting it_tr_que") # Actually this will not be sufficient to avoid assertion error beacuae batch number is 
    it_tr_que.batch_number=0        # already in the prepared batches. However, below I reinitialize it_tr_que. This mess
                                    # should be solved.
    
#############################################################################    
### --- Define loss, train functions etc. ------------------------------- ###
#############################################################################    
        
    loss_  = tf.tensordot(WGT_m_p, tf.nn.sparse_softmax_cross_entropy_with_logits( labels = L_m_p, logits = C_m_ ), axes=[[0], [0]] )
    loss_  = loss_ / np.log(n_spk)                            # Normalize with the loss of random predictions 
    loss_ += model.get_l2_reg(l2_reg_b_pool, l2_reg_a_pool, l2_reg_multi)

    params_to_update_ = model.get_upd_parameters()
    print(params_to_update_)
    grads_, vars_     = list(zip(*optim.compute_gradients(loss = loss_, var_list = params_to_update_, gate_gradients=optim.GATE_GRAPH))) 
    info_             = [ ]  # For debugging something can be added here (e.g. reg_loss, grads) will be printe each epoch

    # Apply Kaldi style gradient repairing
    # TODO: This is hardwired in some other scripts but not added here yet.
    # So far (2018-07-19) this has not improved performace though.
    
    min_op        = optim.apply_gradients(list(zip(grads_,vars_)))
    tr_outputs_   = [info_, loss_, min_op]

    
    def train_function(X1, C1, WGT_m, L_m, lr):
        if not annoying_train:
            tr_inputs_ = {X1_p: X1, C1_p: C1, WGT_m_p: WGT_m, L_m_p: L_m, lr_p: lr, is_test_p: False}
        else:

            tr_inputs_ = {X1_p: X1, WGT_m_p: WGT_m, L_m_p: L_m, lr_p: lr, is_test_p: False, C1_p: np.array([])}
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

    ### --- Remove this way of saving when the below is tested properly
    params_to_store_ = model.get_parameters()
    optim_states_    = get_optim_states(optim)
    def get_para():
        return [sess.run(params_to_store_), sess.run(optim_states_)]

    print(get_para())

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

            
    # This function will be checked after each epoch. Its output will be used
    # to determine whether to half the learning rate.
    # NOTE: If only one utt, C will be a vector instead of  a matrix!!! This means argmax line will not work!!
    def check_dev_multi_loss_acc():
        X1, C1 = load_feats_dev( list(range(len(dev_scp_info['utt2file'])))  )
        L_m    = dev_scp_info['utt2spk']
        WGT_m = np.ones(L_m.shape)/ float(L_m.shape[0])
        L, C = sess.run([loss_, C_m_], {X1_p: X1, C1_p:C1, WGT_m_p:WGT_m, L_m_p:L_m, is_test_p: True} )
        P = np.argmax(C, axis=1)
        Acc = sum(P == L_m)/float(len(L_m))
        log.info("Loss %f, Accuracy: %f", L, Acc )
        return L   

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
        
    check_dev_multi_loss_acc()


    batch_start = 0
    batch_count = batch_start


    # The batch_iterator class prepares batches in a second thread while the training is running.
    all_bad_utts = [] # Keeps track of 0-duration utterances. (Which are ignored in trainin). Not used if annoying_train=True  
    it_tr_que    = utils.mbatch_generation.batch_iterator_2(it_tr, load_feats_train,
                                                            annoying_train, batch_que_length, batch_count, use_mpi=False)

    # Function that will do training on one batch
    train_batch_func = get_train_batch_multi_fkn( it_tr_que, train_function )


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
    train_nn(n_epoch, n_batch, lr_first, lr_last, train_batch_func, check_dev =check_dev_multi_loss_acc, 
             get_para_func =get_para, set_para_func =set_para, model_save_file =model_prefix,
             patience=patience, save_func=save_func, patience_2=patience_2)
 
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




