

from utils.misc import get_logger
log = get_logger()

import numpy as np
import tensorflow as tf

import tensorflow_code.initializers
import tensorflow_code.nn_def

from tensorflow_code.functions import tf_reverse_gradient

class xvector_bn_mlt(object):

    def __init__(self, session, is_test_p, n_spk, tdnn_sizes_before_pooling=None, tdnn_sizes_after_pooling=None,
                 activations_before_pooling=None, activations_after_pooling=None, pool_function=None, pool_size=None,
                 it_tr_que=None, scp_info=None, load_data_function=None, upd_b_pool=True, upd_a_pool=True,
                 upd_multi=True, do_feat_norm=False, upd_feat_norm=False, do_pool_norm=False, upd_pool_norm=False,
                 use_bug=False, floatX='float32'):

        # Perhaps not great to have it here, but layers require it for their get_parameters functions etc.
        # Alternative would be to always pass sessions to these calls or use default session. ... Think about this.
        # Same with the is_test_p variable, Could have been passed with each call to the model but not sure if this
        # is more convenient.
        self.session = session
        self.is_test_p = is_test_p

        # Whether to use the buggy initialization
        self.use_bug=use_bug
        
        # If no architecture is given, we assume the JHU one.
        if tdnn_sizes_before_pooling == None:
            self.tdnn_sizes_before_pooling = ( [115, 512, 5, 1], [1536, 512, 3, 2], [1536,512, 3, 3],
                                                 [512, 512, 1,1], [512,1500,1,1] )
        else:
            self.tdnn_sizes_before_pooling = tdnn_sizes_before_pooling
       
        if tdnn_sizes_after_pooling == None:
            self.tdnn_sizes_after_pooling = ( [3000, 512,1,1], [512, 512,1,1] ) # Actually, these would normally be normal dense layers
        else:                                                                    
            self.tdnn_sizes_after_pooling = tdnn_sizes_after_pooling           

        self.n_lay_b_pool   = len(tdnn_sizes_before_pooling)
        self.feat_dim       = self.tdnn_sizes_before_pooling[0][0] / self.tdnn_sizes_before_pooling[0][2] # Input_dim / n_step
        self.n_lay_a_pool   = len(tdnn_sizes_after_pooling)

        
        if activations_before_pooling == None:        
            self.activations_before_pooling = [tf.nn.relu] * self.n_lay_b_pool
        else:
            self.activations_before_pooling = activations_before_pooling

        if activations_after_pooling  == None:        
            self.activations_after_pooling = [tf.nn.relu] * self.n_lay_a_pool
        else:
            self.activations_after_pooling = activations_after_pooling            

        if pool_function == None:
            self.pool_function = lambda x: tensorflow_code.pool_fkns.mean_std(x, axes=1)
        else:
            self.pool_function = pool_function

        if pool_size == None:
            self.pool_size = self.tdnn_sizes_before_pooling[-1][1] * 2
        else:    
            self.pool_size = pool_size 

        log.info("TDNN architecture before pooling: " + str(list(zip(self.tdnn_sizes_before_pooling,self.activations_before_pooling))) )
        log.info("TDNN architecture after pooling: "  + str(list(zip(self.tdnn_sizes_after_pooling,self.activations_after_pooling))) )       
        
        self.floatX        = floatX
        self.is_test_p     = is_test_p
        self.upd_b_pool    = upd_b_pool
        self.upd_a_pool    = upd_a_pool
        self.upd_multi     = upd_multi
        self.do_feat_norm  = do_feat_norm
        self.upd_feat_norm = upd_feat_norm
        self.do_pool_norm  = do_pool_norm
        self.upd_pool_norm = upd_pool_norm
        #self.it_tr         = it_tr
        self.n_spk         = n_spk
        
        log.info('Initializing model randomly')
        np.random.seed(17)
        #lda_tmp    = tensorflow_code.initializers.init_params_simple_he_uniform( (512, 150),
        #                                                                         floatX=floatX, use_bug=self.use_bug) # Delete this

                
        self.n_lay_b_pool = len(tdnn_sizes_before_pooling)
        self.n_lay_a_pool = len(tdnn_sizes_after_pooling)

        bn_decay = 0.95 # Decay rate for batch-norm.

        self.layers_before_pooling = []            
        self.layers_after_pooling  = []                                               

        ##############################################################################
        ### Estimate normalization of feats if desired.        
        if (do_feat_norm):
            
            if (it_tr_que != None ):
                log.info("Estimating feature normalization")
                #[X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                [X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()


                ### Apply the normalization    
                mean_feat = np.mean(feats[0],axis=(0,1))[np.newaxis,np.newaxis,:]
                var_feat  = np.var(feats[0],axis=(0,1))[np.newaxis,np.newaxis,:]
            else:
                log.info("Feature normalization initialized with mean=0, std=1" )                
                mean_feat = np.zeros([1,1,self.feat_dim])
                var_feat  = np.ones([1,1,self.feat_dim])
                
            if ( self.upd_feat_norm ):
                log.info("Feature norm will be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
            else:
                log.info("Feature norm will not be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
            self.layers_before_pooling +=  [bn_feats] 
    
        ##########################################################################        
        ### Layers before pooling
        if self.upd_b_pool:
            is_test_b_pool = self.is_test_p
        else:
            is_test_b_pool = tf.constant(True)

        for i in range( self.n_lay_b_pool ):
            assert ( len(tdnn_sizes_before_pooling[i])==4 )
                
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_before_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_before_pooling[i][2]
            step_size = tdnn_sizes_before_pooling[i][3]

            self.layers_before_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                             n_step=n_step,
                                                                             in_dim=para['W_1'].shape[0] / n_step,
                                                                             out_dim=para['W_1'].shape[1],
                                                                             step_size=step_size,
                                                                             floatX=self.floatX) )

            # Append the non-linearity
            self.layers_before_pooling.append(activations_before_pooling[i])
            
            # Append Batch-norm layer
            self.layers_before_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                            var=np.ones_like(para['b_1']),
                                                                                            offset=np.zeros_like(para['b_1']),
                                                                                            scale=np.ones_like(para['b_1']),
                                                                                            is_test =is_test_b_pool, decay=bn_decay,
                                                                                            floatX=self.floatX ) )

        ##########################################################################
        ### Add and estimate normalization of output from pooling layer if desired.
        #
        # This is the only "architecture difference" from the Kaldi. In order to
        # reach the same performance as Kaldi, we have to make sure that the
        # output from the pooling layer has mean 0 and standard deviation 1. We 
        # achieve this by estimating the output mean and standard deviation on a
        # few batches here after initializing the model and then use this statatistics
        # for normalization. Most likely, Kaldi's optimizer is more robust to this.
           
        # Placeholder that will be used in initialization
        if (do_pool_norm):

            if (it_tr_que != None ):
                log.info("Estimating statistics of pool output")
                X1_p        = tf.placeholder(self.floatX, shape=[None,None,self.feat_dim], name='X1_p') # Features 
                C1_p        = tf.placeholder(dtype='int32', shape=[None], name='C1_p')                  # Indices
                Y_, _, _, _, _ = self.__call__(X1_p, C1_p,annoying_train=True)                             # The output of the pooling layer.
                g_stat      = lambda X1 : self.session.run(Y_, {X1_p: X1, C1_p: np.array([]), self.is_test_p:False})
                ss          = np.zeros([0,self.pool_size])
                
                self.session.run(tf.global_variables_initializer()) 
                log.info("Calculating mean and standard deviation of pooling output") 
                for i in range(10):
                    [X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                    ss    = np.concatenate([ss,g_stat(feats[0]).squeeze()], axis=0)
                
                ### Apply the normalization as a batchnorm layer   
                mean_pool = np.mean(ss,axis=0)[np.newaxis,np.newaxis,:]
                var_pool  = np.var(ss,axis=0)[np.newaxis,np.newaxis,:]

            else:
                log.info("Pool normalization initialized with mean=0, std=1")
                mean_pool = np.zeros([1,1,self.pool_size])
                var_pool  = np.ones([1,1,self.pool_size])
                
            if ( self.upd_pool_norm ):
                log.info("Pool norm will be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX ) 
            else:
                log.info("Pool norm will not be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX )              
            self.layers_after_pooling.append(bn_pool)
        ##########################################################################        
        ### Layers after pooling
        if self.upd_a_pool:
            is_test_a_pool = self.is_test_p
        else:
            is_test_a_pool = tf.constant(True)

        for i in range( self.n_lay_a_pool ):
            assert ( len(tdnn_sizes_after_pooling[i])==4 )
                
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_after_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_after_pooling[i][2]
            step_size = tdnn_sizes_after_pooling[i][3]
                                          
            self.layers_after_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                            n_step=n_step,
                                                                            in_dim=para['W_1'].shape[0] / n_step,
                                                                            out_dim=para['W_1'].shape[1],
                                                                            step_size=step_size, floatX=self.floatX) )
            # Append the non-linearity
            self.layers_after_pooling.append(activations_after_pooling[i])
            
            # Append Batch-norm layer
            self.layers_after_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                           var=np.ones_like(para['b_1']),
                                                                                           offset=np.zeros_like(para['b_1']),
                                                                                           scale=np.ones_like(para['b_1']),
                                                                                           is_test =is_test_a_pool, decay=bn_decay,
                                                                                           floatX=self.floatX ) )    

        #########################################################################
        #### Multiclass classification.
        params_multi_class = tensorflow_code.initializers.init_params_simple_he_uniform( [tdnn_sizes_after_pooling[-1][1], self.n_spk],
                                                                                         floatX=self.floatX, use_bug=self.use_bug) 
        self.nn_multi_class = tensorflow_code.nn_def.tf_ff_nn(self.session, params_multi_class, floatX=self.floatX )
        
                                              
    def __call__(self,X1_, C1_, annoying_train):

        # Will only return what can be returned based on what is initialized.
        # For example, if layer_after_pooling is not initialized, embeddings
        # will not be returned but stats after pooling will. This is useful
        # when we will initialized the "after pooling norm" since the layers
        # afterwards have not been initialized at that stage.
        
        ##########################################################################
        ### Apply pooling and the layers before it.
        #
        # With annoying_train all utterances are processed in on go before pooling
        # which allows proper batch norm. In testing, we procss them one by one.
        # Training
        if (annoying_train):
            def train_pooling():
                Z_ =  X1_
                for j in range(0, len( self.layers_before_pooling ) ):
                    Z_ = self.layers_before_pooling[j]( Z_ )
                    log.info(str(j))
                    if j == 9:
                        log.info("A" + str(j))                                           
                        batch_n_3_ = Z_ 
                    
                Y_train_ = tensorflow_code.pool_fkns.mean_std(Z_, axes=1)
                return Y_train_, batch_n_3_ 

            def test_pooling():
                nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling, pool_fkn =self.pool_function,
                                                                    output_size=self.pool_size, floatX=self.floatX )
                Y_test_ = nn_pool( X1_, C1_ )
                return Y_test_ ,Y_test_ 

            Y_, batch_n_3_ = tf.cond(self.is_test_p, test_pooling, train_pooling)


            
        # Without annoying_train we process the utterances one by one. This can reduces
        # memory requirements (by copying temporary intermediate results to the CPU RAM),
        # But with this option batch norm will not (yet) work properly in training. Should
        # fixed of-course if batch norm is shown to be useful.
        else:
            log.error("Bottle-neck multitask training is not implemented for variable duration utterance training")
            nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling, pool_fkn =self.pool_function,
                                                                output_size=self.pool_size, floatX=self.floatX )
            Y_ = nn_pool( X1_, C1_ )       
                                              
        # This variable is for stats in case we want to extract it.                                  
        stat_  = tf.squeeze(Y_, axis=[1], name='stats' )

        ##########################################################################                                              
        ### Layers after pooling
        embds_=[]
        for i in range( len(self.layers_after_pooling) ):
            Y_ = self.layers_after_pooling[i]( Y_ )
                                              
            # Extract embeddings after the last and the and second last TDNN
            if isinstance(self.layers_after_pooling[i], tensorflow_code.nn_def.tf_tdnn):
                embds_.append(Y_)

        if (len(embds_) >= 2 ):
            embd_A_ = tf.squeeze(embds_[-2], axis=[1], name='embd_A' )
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        elif (len(embds_) >= 1 ):
            embd_A_ = None
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        else:
            embd_A_ = None
            embd_B_ = None 
            
        ##########################################################################
        ### Predictions
        if hasattr(self, 'nn_multi_class'):
            pred_ = tf.identity(self.nn_multi_class(tf.squeeze( Y_)), name='pred') # The identity is just to add the name.
        else:
            pred_ = None
        return stat_, embd_A_, embd_B_, pred_, batch_n_3_ 


    def get_parameters(self):
        params_ = []
        # if self.upd_b_pool:
        for l in self.layers_before_pooling:          #+ self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):       # Need to check this since e.g.relu doesn't have params
                params_ += l.get_parameters()  

        # if self.upd_a_pool:
        for l in self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):
                params_ += l.get_parameters() 
                    
        # if self.upd_multi:
        params_ +=  self.nn_multi_class.get_parameters() 

                    
        return params_
       
    def get_upd_parameters(self):
        params_ = []
        if self.upd_b_pool:
            for l in self.layers_before_pooling  :
                if (hasattr(l, 'get_upd_parameters')):
                    params_ += l.get_upd_parameters() 

        if self.upd_a_pool:
            for l in self.layers_after_pooling:
                if (hasattr(l, 'get_upd_parameters')):
                    params_ += l.get_upd_parameters() 
                    
        if self.upd_multi:
            params_ +=  self.nn_multi_class.get_upd_parameters() 

        return params_

    
    def get_l2_reg(self, b_pool_reg=0.0, a_pool_reg=0.0, multi_reg=0.0):
        l2_reg = 0.0
        if (b_pool_reg > 0.0):
            for l in self.layers_before_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( b_pool_reg )
        if (a_pool_reg > 0.0):
            for l in self.layers_after_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( a_pool_reg )
        if (multi_reg > 0.0):
                    l2_reg += self.nn_multi_class.get_l2_reg( multi_reg )

        return l2_reg


class xvector(object):

    def __init__(self, session, is_test_p, n_spk, tdnn_sizes_before_pooling=None, tdnn_sizes_after_pooling=None,
                 activations_before_pooling=None, activations_after_pooling=None, pool_function=None, pool_size=None,
                 it_tr_que=None, scp_info=None, load_data_function=None, upd_b_pool=True, upd_a_pool=True,
                 upd_multi=True, do_feat_norm=False, upd_feat_norm=False, do_pool_norm=False, upd_pool_norm=False,
                 upd_b_pool_spec=[], use_bug=False, floatX='float32', stop_grad_ap =-1, rev_grad_ap=-1, pool_l_ctx=0, pool_r_ctx=0):

        # Perhaps not great to have it here, but layers require it for their get_parameters functions etc.
        # Alternative would be to always pass sessions to these calls or use default session. ... Think about this.
        # Same with the is_test_p variable, Could have been passed with each call to the model but not sure if this
        # is more convenient.
        self.session = session
        self.is_test_p = is_test_p

        self.stop_grad_ap = stop_grad_ap 
        self.rev_grad_ap = rev_grad_ap
        self.pool_l_ctx = pool_l_ctx
        self.pool_r_ctx = pool_r_ctx
        # Whether to use the buggy initialization
        self.use_bug=use_bug

        # If no architecture is given, we assume the JHU one.
        if tdnn_sizes_before_pooling == None:
            self.tdnn_sizes_before_pooling = ( [115, 512, 5, 1], [1536, 512, 3, 2], [1536,512, 3, 3],
                                                 [512, 512, 1,1], [512,1500,1,1] )
        else:
            self.tdnn_sizes_before_pooling = tdnn_sizes_before_pooling
       
        if tdnn_sizes_after_pooling == None:
            self.tdnn_sizes_after_pooling = ( [3000, 512,1,1], [512, 512,1,1] ) # Actually, these would normally be normal dense layers
        else:                                                                    
            self.tdnn_sizes_after_pooling = tdnn_sizes_after_pooling           

        self.n_lay_b_pool   = len(tdnn_sizes_before_pooling)
        self.feat_dim       = self.tdnn_sizes_before_pooling[0][0] / self.tdnn_sizes_before_pooling[0][2] # Input_dim / n_step
        self.n_lay_a_pool   = len(tdnn_sizes_after_pooling)

        
        if activations_before_pooling == None:        
            self.activations_before_pooling = [tf.nn.relu] * self.n_lay_b_pool
        else:
            self.activations_before_pooling = activations_before_pooling

        if activations_after_pooling  == None:        
            self.activations_after_pooling = [tf.nn.relu] * self.n_lay_a_pool
        else:
            self.activations_after_pooling = activations_after_pooling            

        if pool_function == None:
            self.pool_function = lambda x: tensorflow_code.pool_fkns.mean_std(x, axes=1)
        else:
            self.pool_function = pool_function

        if pool_size == None:
            self.pool_size = self.tdnn_sizes_before_pooling[-1][1] * 2
        else:    
            self.pool_size = pool_size 

        log.info("TDNN architecture before pooling: " + str(list(zip(self.tdnn_sizes_before_pooling,self.activations_before_pooling))) )
        log.info("TDNN architecture after pooling: "  + str(list(zip(self.tdnn_sizes_after_pooling,self.activations_after_pooling))) )       
        
        self.floatX          = floatX
        self.is_test_p       = is_test_p
        self.upd_b_pool      = upd_b_pool
        self.upd_a_pool      = upd_a_pool
        self.upd_multi       = upd_multi
        self.do_feat_norm    = do_feat_norm
        self.upd_feat_norm   = upd_feat_norm
        self.do_pool_norm    = do_pool_norm
        self.upd_pool_norm   = upd_pool_norm
        #self.it_tr          = it_tr
        self.n_spk           = n_spk
        self.upd_b_pool_spec = upd_b_pool_spec

        assert( (len(self.upd_b_pool_spec) == 0) or  (len(self.upd_b_pool_spec) == self.n_lay_b_pool) ) 
        if (len(self.upd_b_pool_spec) == 0):
            self.upd_b_pool_spec = [True] * self.n_lay_b_pool
        if (not self.upd_b_pool):
            log.warning("Providing upd_b_pool_spec is meaningless if upd_b_pool=False")
            self.upd_b_pool_spec = [False] * self.n_lay_b_pool # Just to be extra sure

        
        log.info('Initializing model randomly')
        np.random.seed(17)
        #lda_tmp    = tensorflow_code.initializers.init_params_simple_he_uniform( (512, 150), floatX=floatX,
        #                                                                         use_bug=self.use_bug) # Delete this

                
        self.n_lay_b_pool = len(tdnn_sizes_before_pooling)
        self.n_lay_a_pool = len(tdnn_sizes_after_pooling)

        bn_decay = 0.95 # Decay rate for batch-norm.

        self.layers_before_pooling     = []
        self.layers_before_pooling_upd = []            
        self.layers_after_pooling      = []                                               

        ##############################################################################
        ### Estimate normalization of feats if desired.        
        if (do_feat_norm):
            
            if (it_tr_que != None ):
                log.info("Estimating feature normalization")
                if not isinstance(it_tr_que, list):
                    [X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                else:
                    assert(len(it_tr_que) ==3)
                    log.info("Using specified X1_p and X1_ in feat stat estimation")
                    X1_p = it_tr_que[0]
                    X1_  = it_tr_que[1]                
                    [X, Y, U], _, [feats_1, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que[2].get_batch()
                    feats = self.session.run(X1_, {X1_p:feats_1})
                    
                ### Apply the normalization    
                mean_feat = np.mean(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
                var_feat  = np.var(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
            else:
                log.info("Feature normalization initialized with mean=0, std=1" )                
                mean_feat = np.zeros([1,1,self.feat_dim])
                var_feat  = np.ones([1,1,self.feat_dim])
                
            if ( self.upd_feat_norm ):
                log.info("Feature norm will be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
                self.layers_before_pooling_upd +=  [ True ] 
            else:
                log.info("Feature norm will not be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
                self.layers_before_pooling_upd +=  [ False ] 
            self.layers_before_pooling     +=  [bn_feats] 

        ##########################################################################        
        ### Layers before pooling

        for i in range( self.n_lay_b_pool ):
            assert ( len(tdnn_sizes_before_pooling[i])==4 )


            if self.upd_b_pool and self.upd_b_pool_spec[i]:
                is_test_b_pool = self.is_test_p
            else:
                is_test_b_pool = tf.constant(True)

            
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_before_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_before_pooling[i][2]
            step_size = tdnn_sizes_before_pooling[i][3]

            self.layers_before_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                             n_step=n_step,
                                                                             in_dim=para['W_1'].shape[0] //  n_step,
                                                                             out_dim=para['W_1'].shape[1],
                                                                             step_size=step_size,
                                                                             floatX=self.floatX) )

            # Append the non-linearity
            self.layers_before_pooling.append(activations_before_pooling[i])
            
            # Append Batch-norm layer
            self.layers_before_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                            var=np.ones_like(para['b_1']),
                                                                                            offset=np.zeros_like(para['b_1']),
                                                                                            scale=np.ones_like(para['b_1']),
                                                                                            is_test =is_test_b_pool, decay=bn_decay,
                                                                                            floatX=self.floatX ) )
            if self.upd_b_pool and self.upd_b_pool_spec[i]:
                self.layers_before_pooling_upd +=  [ True, True, True ]
            else:
                self.layers_before_pooling_upd +=  [ False, False, False ]
                
        ##########################################################################
        ### Add and estimate normalization of output from pooling layer if desired.
        #
        # This is the only "architecture difference" from the Kaldi. In order to
        # reach the same performance as Kaldi, we have to make sure that the
        # output from the pooling layer has mean 0 and standard deviation 1. We 
        # achieve this by estimating the output mean and standard deviation on a
        # few batches here after initializing the model and then use this statatistics
        # for normalization. Most likely, Kaldi's optimizer is more robust to this.
           
        # Placeholder that will be used in initialization
        if (do_pool_norm):

            if (it_tr_que != None ):
                log.info("Estimating statistics of pool output")
                if not isinstance(it_tr_que, list):
                    X1_p        = tf.placeholder(self.floatX, shape=[None,None,self.feat_dim], name='X1_p') # Features 
                    C1_p        = tf.placeholder(dtype='int32', shape=[None], name='C1_p')                  # Indices
                    Y_, _, _, _ = self.__call__(X1_p, C1_p,annoying_train=True)                            # The output of the pooling layer.
                    g_stat      = lambda X1 : self.session.run(Y_, {X1_p: X1, C1_p: np.array([]), self.is_test_p:False})
                else:
                    # This is if some processing of X1 is done before it goes to the network
                    assert(len(it_tr_que) ==3)
                    log.info("Using specified X1_p and X1_ in pool stat estimation")
                    X1_p = it_tr_que[0]
                    X1_  = it_tr_que[1]
                    it_tr_que = it_tr_que[2]
                    C1_p        = tf.placeholder(dtype='int32', shape=[None], name='C1_p')                  # Indices
                    Y_, _, _, _ = self.__call__(X1_, C1_p,annoying_train=True)                            # The output of the pooling layer.
                    g_stat      = lambda X1 : self.session.run(Y_, {X1_p: X1, C1_p: np.array([]), self.is_test_p:False})
                ss          = np.zeros([0,self.pool_size])
                                                               
                self.session.run(tf.global_variables_initializer()) 
                log.info("Calculating mean and standard deviation of pooling output") 
                for i in range(10):
                    [X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                    ss    = np.concatenate([ss,g_stat(feats).squeeze()], axis=0)
                
                ### Apply the normalization as a batchnorm layer   
                mean_pool = np.mean(ss,axis=0)[np.newaxis,np.newaxis,:]
                var_pool  = np.var(ss,axis=0)[np.newaxis,np.newaxis,:]

            else:
                log.info("Pool normalization initialized with mean=0, std=1")
                mean_pool = np.zeros([1,1,self.pool_size])
                var_pool  = np.ones([1,1,self.pool_size])
                
            if ( self.upd_pool_norm ):
                log.info("Pool norm will be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX ) 
            else:
                log.info("Pool norm will not be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX )              
            self.layers_after_pooling.append(bn_pool)
        ##########################################################################        
        ### Layers after pooling
        if self.upd_a_pool:
            is_test_a_pool = self.is_test_p
        else:
            is_test_a_pool = tf.constant(True)

        for i in range( self.n_lay_a_pool ):
            assert ( len(tdnn_sizes_after_pooling[i])==4 )
                
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_after_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_after_pooling[i][2]
            step_size = tdnn_sizes_after_pooling[i][3]
                                          
            self.layers_after_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                            n_step=n_step,
                                                                            in_dim=para['W_1'].shape[0] // n_step,
                                                                            out_dim=para['W_1'].shape[1],
                                                                            step_size=step_size, floatX=self.floatX) )
            # Append the non-linearity
            self.layers_after_pooling.append(activations_after_pooling[i])
            
            # Append Batch-norm layer
            self.layers_after_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                           var=np.ones_like(para['b_1']),
                                                                                           offset=np.zeros_like(para['b_1']),
                                                                                           scale=np.ones_like(para['b_1']),
                                                                                           is_test =is_test_a_pool, decay=bn_decay,
                                                                                           floatX=self.floatX ) )    

        #########################################################################
        #### Multiclass classification.
        params_multi_class = tensorflow_code.initializers.init_params_simple_he_uniform( [tdnn_sizes_after_pooling[-1][1], self.n_spk],
                                                                                         floatX=self.floatX, use_bug=self.use_bug) 
        self.nn_multi_class = tensorflow_code.nn_def.tf_ff_nn(self.session, params_multi_class, floatX=self.floatX )
        
                                              
    def __call__(self,X1_, C1_, annoying_train, lay_b_pool_return=[], extra_lay_before_b_pool=[], embd_A_idx=-2):

        # Will only return what can be returned based on what is initialized.
        # For example, if layer_after_pooling is not initialized, embeddings
        # will not be returned but stats after pooling will. This is useful
        # when we will initialized the "after pooling norm" since the layers
        # afterwards have not been initialized at that stage.
        
        ##########################################################################
        ### Apply pooling and the layers before it.
        #
        # With annoying_train all utterances are processed in one go before pooling
        # which allows proper batch norm. In testing, we procss them one by one.
        # Training
        all_lay_before_pool = extra_lay_before_b_pool + self.layers_before_pooling
        if (annoying_train):
            def train_pooling():
                Z_ =  X1_
                # for j in range(0, len( self.layers_before_pooling ) ):
                #    Z_ = self.layers_before_pooling[j]( Z_ )
                for j in range(0, len( all_lay_before_pool ) ):
                    Z_ = all_lay_before_pool[j]( Z_ )
                    log.info("Applying %s", str(Z_))
                    
                Y_train_ = tensorflow_code.pool_fkns.mean_std(Z_, axes=1)
                return Y_train_

            def test_pooling():
                #nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling, pool_fkn =self.pool_function,
                #                                              output_size=self.pool_size, floatX=self.floatX,
                #                                              l_ctx=self.pool_l_ctx,r_ctx=self.pool_r_ctx )
                nn_pool     = tensorflow_code.nn_def.tf_pool( all_lay_before_pool, pool_fkn =self.pool_function,
                                                              output_size=self.pool_size, floatX=self.floatX,
                                                              l_ctx=self.pool_l_ctx,r_ctx=self.pool_r_ctx )

                Y_test_ = nn_pool( X1_, C1_ )
                return Y_test_

            Y_ = tf.cond(self.is_test_p, test_pooling, train_pooling)

        # Without annoying_train we process the utterances one by one. This can reduces
        # memory requirements (by copying temporary intermediate results to the CPU RAM),
        # But with this option batch norm will not (yet) work properly in training. Should
        # fixed of-course if batch norm is shown to be useful.
        else:
            #nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling, pool_fkn =self.pool_function,
            #                                              output_size=self.pool_size, loop_swap_memory=True, floatX=self.floatX,
            #                                              l_ctx=self.pool_l_ctx, r_ctx=self.pool_r_ctx)
            nn_pool     = tensorflow_code.nn_def.tf_pool( all_lay_before_pool, pool_fkn =self.pool_function,
                                                          output_size=self.pool_size, loop_swap_memory=True, floatX=self.floatX,
                                                          l_ctx=self.pool_l_ctx, r_ctx=self.pool_r_ctx)
            Y_ = nn_pool( X1_, C1_ )       
                                              
        # This variable is for stats in case we want to extract it.                                  
        stat_  = tf.squeeze(Y_, axis=[1], name='stats' )

        b_pool_extra_out_ = []
        ZE_ =  X1_        
        for j in range(0, len( self.layers_before_pooling ) ):
            ZE_ = self.layers_before_pooling[j]( ZE_ )
            if j in lay_b_pool_return:
                b_pool_extra_out_.append(ZE_)
                log.info("Output also %s",  str(ZE_))
                log.info(ZE_)
        assert( len(lay_b_pool_return) == len(b_pool_extra_out_) )
                
        ##########################################################################                                              
        ### Layers after pooling
        embds_=[]
        for i in range( len(self.layers_after_pooling) ):
            if (i == self.stop_grad_ap ):
                log.info("Stopping gradient between between (any rev here will be ignored)" + str(Y_))
                Y_ = self.layers_after_pooling[i]( tf.stop_gradient(Y_) ) # Gradiends will be stopped between layer i and i-1 
                log.info("and " + str(Y_))
            elif (i == self.rev_grad_ap ):
                log.info("Reversing gradient between between" + str(Y_))
                Y_ = self.layers_after_pooling[i](  tf_reverse_gradient(Y_) ) # Gradiends will be stopped between layer i and i-1 
                log.info("and " + str(Y_))
            else:
                Y_ = self.layers_after_pooling[i]( Y_ )
                
            # Extract embeddings after the last and the and second last TDNN
            if isinstance(self.layers_after_pooling[i], tensorflow_code.nn_def.tf_tdnn):
                embds_.append(Y_)

        if (len(embds_) >= 2 ):
            embd_A_ = tf.squeeze(embds_[embd_A_idx], axis=[1], name='embd_A' )
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        elif (len(embds_) >= 1 ):
            embd_A_ = None
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        else:
            embd_A_ = None
            embd_B_ = None 
            
        ##########################################################################
        ### Predictions
        if hasattr(self, 'nn_multi_class'):
            pred_ = tf.identity(self.nn_multi_class(tf.squeeze( Y_)), name='pred') # The identity is just to add the name.
        else:
            pred_ = None

        if len(b_pool_extra_out_) > 0:
            return stat_, embd_A_, embd_B_, pred_, b_pool_extra_out_
        else:
            return stat_, embd_A_, embd_B_, pred_ 


    def get_parameters(self):
        params_ = []
        #if self.upd_b_pool:
        for l in self.layers_before_pooling:          #+ self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):       # Need to check this since e.g.relu doesn't have params
                params_ += l.get_parameters()  

        #if self.upd_a_pool:
        for l in self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):
                params_ += l.get_parameters() 
                    
        #if self.upd_multi:
        params_ +=  self.nn_multi_class.get_parameters() 

                    
        return params_
       
    def get_upd_parameters(self):
        params_ = []
        if self.upd_b_pool:
            for i,l in enumerate(self.layers_before_pooling)  :
                #if (hasattr(l, 'get_upd_parameters')):
                if (hasattr(l, 'get_upd_parameters')) and self.layers_before_pooling_upd[i]:
                    params_ += l.get_upd_parameters() 

        if self.upd_a_pool:
            for l in self.layers_after_pooling:
                if (hasattr(l, 'get_upd_parameters')):
                    params_ += l.get_upd_parameters() 
                    
        if self.upd_multi:
            params_ +=  self.nn_multi_class.get_upd_parameters() 

        return params_

    
    def get_l2_reg(self, b_pool_reg=0.0, a_pool_reg=0.0, multi_reg=0.0):
        l2_reg = 0.0
        if (b_pool_reg > 0.0):
            for l in self.layers_before_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( b_pool_reg )
        if (a_pool_reg > 0.0):
            for l in self.layers_after_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( a_pool_reg )
        if (multi_reg > 0.0):
                    l2_reg += self.nn_multi_class.get_l2_reg( multi_reg )

        return l2_reg



class xvector_residual(object):

    def __init__(self, session, is_test_p, n_spk, tdnn_sizes_before_pooling=None, tdnn_sizes_after_pooling=None,
                 activations_before_pooling=None, activations_after_pooling=None, pool_function=None, pool_size=None,
                 it_tr_que=None, scp_info=None, load_data_function=None, upd_b_pool=True, upd_a_pool=True,
                 upd_multi=True, do_feat_norm=False, upd_feat_norm=False, do_pool_norm=False, upd_pool_norm=False,
                 upd_b_pool_spec=[], use_bug=False, floatX='float32', stop_grad_ap =-1, residual_connections_bp=[], residual_ap=[] ):

        # Perhaps not great to have it here, but layers require it for their get_parameters functions etc.
        # Alternative would be to always pass sessions to these calls or use default session. ... Think about this.
        # Same with the is_test_p variable, Could have been passed with each call to the model but not sure if this
        # is more convenient.
        self.session = session
        self.is_test_p = is_test_p

        self.residual_connections_bp = residual_connections_bp;
        self.outputs_to_keep      = np.unique( [i for l in self.residual_connections_bp for i in l ] )
        self.stop_grad_ap = stop_grad_ap 

        self.residual_ap = residual_ap
        
        # Whether to use the buggy initialization
        self.use_bug=use_bug

        # If no architecture is given, we assume the JHU one.
        if tdnn_sizes_before_pooling == None:
            self.tdnn_sizes_before_pooling = ( [115, 512, 5, 1], [1536, 512, 3, 2], [1536,512, 3, 3],
                                                 [512, 512, 1,1], [512,1500,1,1] )
        else:
            self.tdnn_sizes_before_pooling = tdnn_sizes_before_pooling
       
        if tdnn_sizes_after_pooling == None:
            self.tdnn_sizes_after_pooling = ( [3000, 512,1,1], [512, 512,1,1] ) # Actually, these would normally be normal dense layers
        else:                                                                    
            self.tdnn_sizes_after_pooling = tdnn_sizes_after_pooling           

        self.n_lay_b_pool   = len(tdnn_sizes_before_pooling)
        self.feat_dim       = self.tdnn_sizes_before_pooling[0][0] / self.tdnn_sizes_before_pooling[0][2] # Input_dim / n_step
        self.n_lay_a_pool   = len(tdnn_sizes_after_pooling)

        
        if activations_before_pooling == None:        
            self.activations_before_pooling = [tf.nn.relu] * self.n_lay_b_pool
        else:
            self.activations_before_pooling = activations_before_pooling

        if activations_after_pooling  == None:        
            self.activations_after_pooling = [tf.nn.relu] * self.n_lay_a_pool
        else:
            self.activations_after_pooling = activations_after_pooling            

        if pool_function == None:
            self.pool_function = lambda x: tensorflow_code.pool_fkns.mean_std(x, axes=1)
        else:
            self.pool_function = pool_function

        if pool_size == None:
            self.pool_size = self.tdnn_sizes_before_pooling[-1][1] * 2
        else:    
            self.pool_size = pool_size 

        log.info("TDNN architecture before pooling: " + str(list(zip(self.tdnn_sizes_before_pooling,self.activations_before_pooling))) )
        log.info("TDNN architecture after pooling: "  + str(list(zip(self.tdnn_sizes_after_pooling,self.activations_after_pooling))) )       
        
        self.floatX          = floatX
        self.is_test_p       = is_test_p
        self.upd_b_pool      = upd_b_pool
        self.upd_a_pool      = upd_a_pool
        self.upd_multi       = upd_multi
        self.do_feat_norm    = do_feat_norm
        self.upd_feat_norm   = upd_feat_norm
        self.do_pool_norm    = do_pool_norm
        self.upd_pool_norm   = upd_pool_norm
        #self.it_tr          = it_tr
        self.n_spk           = n_spk
        self.upd_b_pool_spec = upd_b_pool_spec

        assert( (len(self.upd_b_pool_spec) == 0) or  (len(self.upd_b_pool_spec) == self.n_lay_b_pool) ) 
        if (len(self.upd_b_pool_spec) == 0):
            self.upd_b_pool_spec = [True] * self.n_lay_b_pool
        if (not self.upd_b_pool):
            log.warning("Providing upd_b_pool_spec is meaningless if upd_b_pool=False")
            self.upd_b_pool_spec = [False] * self.n_lay_b_pool # Just to be extra sure

        
        log.info('Initializing model randomly')
        np.random.seed(17)
        #lda_tmp    = tensorflow_code.initializers.init_params_simple_he_uniform( (512, 150), floatX=floatX,
        #                                                                         use_bug=self.use_bug) # Delete this

                
        self.n_lay_b_pool = len(tdnn_sizes_before_pooling)
        self.n_lay_a_pool = len(tdnn_sizes_after_pooling)

        bn_decay = 0.95 # Decay rate for batch-norm.

        self.layers_before_pooling     = []
        self.layers_before_pooling_upd = []            
        self.layers_after_pooling      = []                                               

        ##############################################################################
        ### Estimate normalization of feats if desired.        
        if (do_feat_norm):
            
            if (it_tr_que != None ):
                log.info("Estimating feature normalization")
                if not isinstance(it_tr_que, list):
                    [X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                else:
                    assert(len(it_tr_que) ==3)
                    log.info("Using specified X1_p and X1_ in feat stat estimation")
                    X1_p = it_tr_que[0]
                    X1_  = it_tr_que[1]                
                    [X, Y, U], _, [feats_1, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que[2].get_batch()
                    feats = self.session.run(X1_, {X1_p:feats_1})
                    
                ### Apply the normalization    
                mean_feat = np.mean(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
                var_feat  = np.var(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
            else:
                log.info("Feature normalization initialized with mean=0, std=1" )                
                mean_feat = np.zeros([1,1,self.feat_dim])
                var_feat  = np.ones([1,1,self.feat_dim])
                
            if ( self.upd_feat_norm ):
                log.info("Feature norm will be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
                self.layers_before_pooling_upd +=  [ True ] 
            else:
                log.info("Feature norm will not be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
                self.layers_before_pooling_upd +=  [ False ] 
            self.layers_before_pooling     +=  [bn_feats] 

        ##########################################################################        
        ### Layers before pooling

        for i in range( self.n_lay_b_pool ):
            assert ( len(tdnn_sizes_before_pooling[i])==4 )


            if self.upd_b_pool and self.upd_b_pool_spec[i]:
                is_test_b_pool = self.is_test_p
            else:
                is_test_b_pool = tf.constant(True)

            
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_before_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_before_pooling[i][2]
            step_size = tdnn_sizes_before_pooling[i][3]

            self.layers_before_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                             n_step=n_step,
                                                                             in_dim=para['W_1'].shape[0] //  n_step,
                                                                             out_dim=para['W_1'].shape[1],
                                                                             step_size=step_size,
                                                                             floatX=self.floatX) )

            # Append the non-linearity
            self.layers_before_pooling.append(activations_before_pooling[i])
            
            # Append Batch-norm layer
            self.layers_before_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                            var=np.ones_like(para['b_1']),
                                                                                            offset=np.zeros_like(para['b_1']),
                                                                                            scale=np.ones_like(para['b_1']),
                                                                                            is_test =is_test_b_pool, decay=bn_decay,
                                                                                            floatX=self.floatX ) )
            if self.upd_b_pool and self.upd_b_pool_spec[i]:
                self.layers_before_pooling_upd +=  [ True, True, True ]
            else:
                self.layers_before_pooling_upd +=  [ False, False, False ]
                
        ##########################################################################
        ### Add and estimate normalization of output from pooling layer if desired.
        #
        # This is the only "architecture difference" from the Kaldi. In order to
        # reach the same performance as Kaldi, we have to make sure that the
        # output from the pooling layer has mean 0 and standard deviation 1. We 
        # achieve this by estimating the output mean and standard deviation on a
        # few batches here after initializing the model and then use this statatistics
        # for normalization. Most likely, Kaldi's optimizer is more robust to this.
           
        # Placeholder that will be used in initialization
        if (do_pool_norm):

            if (it_tr_que != None ):
                log.info("Estimating statistics of pool output")
                if not isinstance(it_tr_que, list):
                    X1_p        = tf.placeholder(self.floatX, shape=[None,None,self.feat_dim], name='X1_p') # Features 
                    C1_p        = tf.placeholder(dtype='int32', shape=[None], name='C1_p')                  # Indices
                    Y_, _, _, _ = self.__call__(X1_p, C1_p,annoying_train=True)                            # The output of the pooling layer.
                    g_stat      = lambda X1 : self.session.run(Y_, {X1_p: X1, C1_p: np.array([]), self.is_test_p:False})
                else:
                    # This is if some processing of X1 is done before it goes to the network
                    assert(len(it_tr_que) ==3)
                    log.info("Using specified X1_p and X1_ in pool stat estimation")
                    X1_p = it_tr_que[0]
                    X1_  = it_tr_que[1]
                    it_tr_que = it_tr_que[2]
                    C1_p        = tf.placeholder(dtype='int32', shape=[None], name='C1_p')                  # Indices
                    Y_, _, _, _ = self.__call__(X1_, C1_p,annoying_train=True)                            # The output of the pooling layer.
                    g_stat      = lambda X1 : self.session.run(Y_, {X1_p: X1, C1_p: np.array([]), self.is_test_p:False})
                ss          = np.zeros([0,self.pool_size])
                                                               
                self.session.run(tf.global_variables_initializer()) 
                log.info("Calculating mean and standard deviation of pooling output") 
                for i in range(10):
                    [X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                    ss    = np.concatenate([ss,g_stat(feats).squeeze()], axis=0)
                
                ### Apply the normalization as a batchnorm layer   
                mean_pool = np.mean(ss,axis=0)[np.newaxis,np.newaxis,:]
                var_pool  = np.var(ss,axis=0)[np.newaxis,np.newaxis,:]

            else:
                log.info("Pool normalization initialized with mean=0, std=1")
                mean_pool = np.zeros([1,1,self.pool_size])
                var_pool  = np.ones([1,1,self.pool_size])
                
            if ( self.upd_pool_norm ):
                log.info("Pool norm will be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX ) 
            else:
                log.info("Pool norm will not be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX )              
            self.layers_after_pooling.append(bn_pool)
        ##########################################################################        
        ### Layers after pooling
        if self.upd_a_pool:
            is_test_a_pool = self.is_test_p
        else:
            is_test_a_pool = tf.constant(True)

        for i in range( self.n_lay_a_pool ):
            assert ( len(tdnn_sizes_after_pooling[i])==4 )
                
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_after_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_after_pooling[i][2]
            step_size = tdnn_sizes_after_pooling[i][3]
                                          
            self.layers_after_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                            n_step=n_step,
                                                                            in_dim=para['W_1'].shape[0] // n_step,
                                                                            out_dim=para['W_1'].shape[1],
                                                                            step_size=step_size, floatX=self.floatX) )
            # Append the non-linearity
            self.layers_after_pooling.append(activations_after_pooling[i])
            
            # Append Batch-norm layer
            self.layers_after_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                           var=np.ones_like(para['b_1']),
                                                                                           offset=np.zeros_like(para['b_1']),
                                                                                           scale=np.ones_like(para['b_1']),
                                                                                           is_test =is_test_a_pool, decay=bn_decay,
                                                                                           floatX=self.floatX ) )    

        #########################################################################
        #### Multiclass classification.
        params_multi_class = tensorflow_code.initializers.init_params_simple_he_uniform( [tdnn_sizes_after_pooling[-1][1], self.n_spk],
                                                                                         floatX=self.floatX, use_bug=self.use_bug) 
        self.nn_multi_class = tensorflow_code.nn_def.tf_ff_nn(self.session, params_multi_class, floatX=self.floatX )
        
                                              
    def __call__(self,X1_, C1_, annoying_train):

        if len(self.residual_connections_bp) != 0:
            log.info("Residual connections " + str( self.residual_connections_bp ) )
            outputs_to_keep = np.unique( [i for l in self.residual_connections_bp for i in l ] )
            kept_outputs = {}

        # Will only return what can be returned based on what is initialized.
        # For example, if layer_after_pooling is not initialized, embeddings
        # will not be returned but stats after pooling will. This is useful
        # when we will initialized the "after pooling norm" since the layers
        # afterwards have not been initialized at that stage.
        
        ##########################################################################
        ### Apply pooling and the layers before it.
        #
        # With annoying_train all utterances are processed in on go before pooling
        # which allows proper batch norm. In testing, we procss them one by one.
        # Training
        if (annoying_train):
            def train_pooling():
                Z_ =  X1_
                for j in range(0, len( self.layers_before_pooling ) ):
                    if ( len(self.residual_connections_bp) > 0):

                        log.debug("Layer " + str(j) + " " + str(self.layers_before_pooling[j]) )
                        log.debug("Residuals  " + str(self.residual_connections_bp[j]) )
                        for o in self.residual_connections_bp[j]:
                            log.debug("   Adding " +str(o) + " " + str( kept_outputs[ o ] ) )

                            sh11_ = tf.shape(kept_outputs[ o ])[1]
                            sh21_ = tf.shape(Z_ )[1]
                            d1_   = tf.cast((sh11_ - sh21_)/2,"int32")
                            Z_r_ = kept_outputs[ o ][:, d1_:sh11_ - d1_,: ]
                            Z_ += Z_r_    #kept_outputs[ o ][d_:-d_]

                    Z_ = self.layers_before_pooling[j]( Z_ )

                    if j in outputs_to_keep:
                        log.debug( "Layer " + str(j) + " " + str(self.layers_before_pooling[j]) + " will be stored. " )
                        kept_outputs[j] = Z_

                Y_train_ = tensorflow_code.pool_fkns.mean_std(Z_, axes=1)
                return Y_train_

            def test_pooling():
                nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling, pool_fkn =self.pool_function,
                                                                    output_size=self.pool_size, floatX=self.floatX )
                Y_test_ = nn_pool( X1_, C1_, res_con=self.residual_connections_bp )
                return Y_test_

            Y_ = tf.cond(self.is_test_p, test_pooling, train_pooling)

        # Without annoying_train we process the utterances one by one. This can reduces
        # memory requirements (by copying temporary intermediate results to the CPU RAM),
        # But with this option batch norm will not (yet) work properly in training. Should
        # fixed of-course if batch norm is shown to be useful.
        else:
            nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling, pool_fkn =self.pool_function,
                                                                output_size=self.pool_size, loop_swap_memory=True, floatX=self.floatX )
            Y_ = nn_pool( X1_, C1_, res_con=self.residual_connections_bp )       
                                              
        # This variable is for stats in case we want to extract it.                                  
        stat_  = tf.squeeze(Y_, axis=[1], name='stats' )

        ##########################################################################                                              
        ### Layers after pooling
        embds_=[]
        res_ap_ = []
        print(len(self.layers_after_pooling))
        for i in range( len(self.layers_after_pooling) ):
            print(i)
            print(self.layers_after_pooling[i])
            if (i == self.stop_grad_ap ):
                log.info("Stopping gradient between between" + str(Y_))
                Y_ = self.layers_after_pooling[i]( tf.stop_gradient(Y_) ) # Gradiends will be stopped between layer i and i-1 
                log.info("and " + str(Y_))
            else:
                Y_ = self.layers_after_pooling[i]( Y_ )
                
            # Extract embeddings after the last and the and second last TDNN
            if isinstance(self.layers_after_pooling[i], tensorflow_code.nn_def.tf_tdnn):
                embds_.append(Y_)

                
            if isinstance(self.layers_after_pooling[i], tensorflow_code.nn_def.tf_batch_norm_mov_avg):
                res_ap_.append(Y_)
        if (len(self.layers_after_pooling)):
            print("Q#")
            print(len(res_ap_))
            print(len(self.residual_ap))
            for i in range(len(self.residual_ap)):
                print(i)
                print(self.residual_ap[i])

                log.debug("Adding residual to softmax: " + str(res_ap_[self.residual_ap[i]]) )
                Y_ += res_ap_[self.residual_ap[i]]
            
        if (len(embds_) >= 2 ):
            embd_A_ = tf.squeeze(embds_[-2], axis=[1], name='embd_A' )
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        elif (len(embds_) >= 1 ):
            embd_A_ = None
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        else:
            embd_A_ = None
            embd_B_ = None 
            
        ##########################################################################
        ### Predictions
        if hasattr(self, 'nn_multi_class'):
            pred_ = tf.identity(self.nn_multi_class(tf.squeeze( Y_)), name='pred') # The identity is just to add the name.
        else:
            pred_ = None
        return stat_, embd_A_, embd_B_, pred_ 


    def get_parameters(self):
        params_ = []
        #if self.upd_b_pool:
        for l in self.layers_before_pooling:          #+ self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):       # Need to check this since e.g.relu doesn't have params
                params_ += l.get_parameters()  

        #if self.upd_a_pool:
        for l in self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):
                params_ += l.get_parameters() 
                    
        #if self.upd_multi:
        params_ +=  self.nn_multi_class.get_parameters() 

                    
        return params_
       
    def get_upd_parameters(self):
        params_ = []
        if self.upd_b_pool:
            for i,l in enumerate(self.layers_before_pooling)  :
                #if (hasattr(l, 'get_upd_parameters')):
                if (hasattr(l, 'get_upd_parameters')) and self.layers_before_pooling_upd[i]:
                    params_ += l.get_upd_parameters() 

        if self.upd_a_pool:
            for l in self.layers_after_pooling:
                if (hasattr(l, 'get_upd_parameters')):
                    params_ += l.get_upd_parameters() 
                    
        if self.upd_multi:
            params_ +=  self.nn_multi_class.get_upd_parameters() 

        return params_

    
    def get_l2_reg(self, b_pool_reg=0.0, a_pool_reg=0.0, multi_reg=0.0):
        l2_reg = 0.0
        if (b_pool_reg > 0.0):
            for l in self.layers_before_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( b_pool_reg )
        if (a_pool_reg > 0.0):
            for l in self.layers_after_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( a_pool_reg )
        if (multi_reg > 0.0):
                    l2_reg += self.nn_multi_class.get_l2_reg( multi_reg )

        return l2_reg

###
class xvector_side_info_residual(object):

   
    def __init__(self, session, is_test_p, n_spk, tdnn_sizes_before_pooling=None, tdnn_sizes_after_pooling=None,
                 activations_before_pooling=None, activations_after_pooling=None, pool_function=None, pool_size=None,
                 it_tr_que=None, scp_info=None, load_data_function=None, upd_b_pool=True, upd_a_pool=True,
                 upd_multi=True, do_feat_norm=False, upd_feat_norm=False, do_pool_norm=False, upd_pool_norm=False,
                 side_info_sizes=[], upd_b_pool_spec=[], use_bug=False, floatX='float32', stop_grad_ap =-1,
                 residual_connections_bp=[], residual_ap=[] ):

        # Perhaps not great to have it here, but layers require it for their get_parameters functions etc.
        # Alternative would be to always pass sessions to these calls or use default session. ... Think about this.
        # Same with the is_test_p variable, Could have been passed with each call to the model but not sure if this
        # is more convenient.
        self.session = session
        self.is_test_p = is_test_p

        self.residual_connections_bp = residual_connections_bp;
        self.outputs_to_keep      = np.unique( [i for l in self.residual_connections_bp for i in l ] )
        self.stop_grad_ap = stop_grad_ap 

        self.residual_ap = residual_ap

        
        # Whether to use the buggy initialization
        self.use_bug=use_bug

        # The size of the side input to each tdnn layer
        self.side_info_size = side_info_sizes
        if len(self.side_info_size) == 0:
            self.side_info_size = [0] * (len(tdnn_sizes_before_pooling) + len(tdnn_sizes_after_pooling))
        assert(len(self.side_info_size) == (len(tdnn_sizes_before_pooling) + len(tdnn_sizes_after_pooling)) )

        # If no architecture is given, we assume the JHU one.
        if tdnn_sizes_before_pooling == None:
            self.tdnn_sizes_before_pooling = ( [115, 512, 5, 1], [1536, 512, 3, 2], [1536,512, 3, 3],
                                                 [512, 512, 1,1], [512,1500,1,1] )
        else:
            self.tdnn_sizes_before_pooling = tdnn_sizes_before_pooling
    
        if tdnn_sizes_after_pooling == None:
            self.tdnn_sizes_after_pooling = ( [3000, 512,1,1], [512, 512,1,1] ) # Actually, these would normally be normal dense layers
        else:                                                                    
            self.tdnn_sizes_after_pooling = tdnn_sizes_after_pooling           

        
        self.n_lay_b_pool   = len(tdnn_sizes_before_pooling)
        self.feat_dim       = self.tdnn_sizes_before_pooling[0][0] / self.tdnn_sizes_before_pooling[0][2] # Input_dim / n_step
        self.n_lay_a_pool   = len(tdnn_sizes_after_pooling)

    
        # Adjust the tdnn size to include th side info
        for i in range( self.n_lay_b_pool ):
            self.tdnn_sizes_before_pooling[i][0] += self.side_info_size[i]
        for i in range( self.n_lay_a_pool ):
            self.tdnn_sizes_after_pooling[i][0] += self.side_info_size[ i + self.n_lay_b_pool ]
            
        if activations_before_pooling == None:        
            self.activations_before_pooling = [tf.nn.relu] * self.n_lay_b_pool
        else:
            self.activations_before_pooling = activations_before_pooling

        if activations_after_pooling  == None:        
            self.activations_after_pooling = [tf.nn.relu] * self.n_lay_a_pool
        else:
            self.activations_after_pooling = activations_after_pooling            

        if pool_function == None:
            self.pool_function = lambda x: tensorflow_code.pool_fkns.mean_std(x, axes=1)
        else:
            self.pool_function = pool_function

        if pool_size == None:
            self.pool_size = self.tdnn_sizes_before_pooling[-1][1] * 2
        else:    
            self.pool_size = pool_size 

        log.info("TDNN architecture before pooling: " + str(list(zip(self.tdnn_sizes_before_pooling,self.activations_before_pooling))) )
        log.info("TDNN architecture after pooling: "  + str(list(zip(self.tdnn_sizes_after_pooling,self.activations_after_pooling))) )       
        
        self.floatX        = floatX
        self.is_test_p     = is_test_p
        self.upd_b_pool    = upd_b_pool
        self.upd_a_pool    = upd_a_pool
        self.upd_multi     = upd_multi
        self.do_feat_norm  = do_feat_norm
        self.upd_feat_norm = upd_feat_norm
        self.do_pool_norm  = do_pool_norm
        self.upd_pool_norm = upd_pool_norm
        #self.it_tr         = it_tr
        self.n_spk         = n_spk

        self.upd_b_pool_spec = upd_b_pool_spec

        assert( (len(self.upd_b_pool_spec) == 0) or  (len(self.upd_b_pool_spec) == self.n_lay_b_pool) ) 
        if (len(self.upd_b_pool_spec) == 0):
            self.upd_b_pool_spec = [True] * self.n_lay_b_pool
            if (not self.upd_b_pool):
                log.warning("Providing upd_b_pool_spec is meaningless if upd_b_pool=False")
                self.upd_b_pool_spec = [False] * self.n_lay_b_pool # Just to be extra sure

        
        log.info('Initializing model randomly')
        np.random.seed(17)
        #lda_tmp    = tensorflow_code.initializers.init_params_simple_he_uniform( (512, 150),
        #                                                                         floatX=floatX, use_bug=self.use_bug) # Delete this

                
        self.n_lay_b_pool = len(tdnn_sizes_before_pooling)
        self.n_lay_a_pool = len(tdnn_sizes_after_pooling)

        bn_decay = 0.95 # Decay rate for batch-norm.

        self.layers_before_pooling = []            
        self.layers_after_pooling  = []                                               
        self.layers_before_pooling_upd = []
        ##############################################################################
        ### Estimate normalization of feats if desired.        
        if (do_feat_norm):
            
            if (it_tr_que != None ):
                log.info("Estimating feature normalization")
                [X, Y, U, S], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()

                ### Apply the normalization    
                mean_feat = np.mean(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
                var_feat  = np.var(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
            else:
                log.info("Feature normalization initialized with mean=0, std=1" )                
                mean_feat = np.zeros([1,1,self.feat_dim])
                var_feat  = np.ones([1,1,self.feat_dim])
                
            if ( self.upd_feat_norm ):
                log.info("Feature norm will be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
                self.layers_before_pooling_upd +=  [ True ] 
            else:
                log.info("Feature norm will not be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
                self.layers_before_pooling_upd +=  [ False ] 
            self.layers_before_pooling +=  [bn_feats] 
    
        ##########################################################################        
        ### Layers before pooling

        for i in range( self.n_lay_b_pool ):
            assert ( len(tdnn_sizes_before_pooling[i])==4 )

            if self.upd_b_pool and self.upd_b_pool_spec[i]:
                is_test_b_pool = self.is_test_p
            else:
                is_test_b_pool = tf.constant(True)
            
                
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_before_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_before_pooling[i][2]
            step_size = tdnn_sizes_before_pooling[i][3]

            self.layers_before_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                             n_step=n_step,
                                                                             in_dim=( para['W_1'].shape[0] - self.side_info_size[i] ) // n_step, 
                                                                             out_dim=para['W_1'].shape[1],
                                                                             step_size=step_size,
                                                                             floatX=self.floatX, side_info_size=self.side_info_size[i]) )


            # Append the non-linearity
            self.layers_before_pooling.append(activations_before_pooling[i])
            
            # Append Batch-norm layer
            self.layers_before_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                            var=np.ones_like(para['b_1']),
                                                                                            offset=np.zeros_like(para['b_1']),
                                                                                            scale=np.ones_like(para['b_1']),
                                                                                            is_test =is_test_b_pool, decay=bn_decay,
                                                                                            floatX=self.floatX ) )
            if self.upd_b_pool and self.upd_b_pool_spec[i]:
                self.layers_before_pooling_upd +=  [ True, True, True ]
            else:
                self.layers_before_pooling_upd +=  [ False, False, False ]
            
        ##########################################################################
        ### Add and estimate normalization of output from pooling layer if desired.
        #
        # This is the only "architecture difference" from the Kaldi. In order to
        # reach the same performance as Kaldi, we have to make sure that the
        # output from the pooling layer has mean 0 and standard deviation 1. We 
        # achieve this by estimating the output mean and standard deviation on a
        # few batches here after initializing the model and then use this statatistics
        # for normalization. Most likely, Kaldi's optimizer is more robust to this.
           
        # Placeholder that will be used in initialization
        if (do_pool_norm):

            if (it_tr_que != None ):
                log.info("Estimating statistics of pool output")
                X1_p        = tf.placeholder(self.floatX, shape=[None,None,self.feat_dim], name='X1_p') # Features 
                C1_p        = tf.placeholder(dtype='int32', shape=[None], name='C1_p')                  # Indices
                S1_p        = tf.placeholder(dtype='int32', shape=[None,1], name='S1_p')                  # Indices
                Y_, _, _, _ = self.__call__(X1_p, C1_p, S1_p, annoying_train=True)                             # The output of the pooling layer.
                g_stat      = lambda X1, S1 : self.session.run(Y_, {X1_p: X1, C1_p: np.array([]), S1_p: S1, self.is_test_p:False})
                ss          = np.zeros([0,self.pool_size])
                
                self.session.run(tf.global_variables_initializer()) 
                log.info("Calculating mean and standard deviation of pooling output") 
                for i in range(10):
                    #[X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                    [X, Y, U, S], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                    ss    = np.concatenate([ss,g_stat(feats, np.vstack(S)).squeeze()], axis=0)
                
                ### Apply the normalization as a batchnorm layer   
                mean_pool = np.mean(ss,axis=0)[np.newaxis,np.newaxis,:]
                var_pool  = np.var(ss,axis=0)[np.newaxis,np.newaxis,:]

            else:
                log.info("Pool normalization initialized with mean=0, std=1")
                mean_pool = np.zeros([1,1,self.pool_size])
                var_pool  = np.ones([1,1,self.pool_size])
                
            if ( self.upd_pool_norm ):
                log.info("Pool norm will be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX ) 
            else:
                log.info("Pool norm will not be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX )              
            self.layers_after_pooling.append(bn_pool)
        ##########################################################################        
        ### Layers after pooling
        if self.upd_a_pool:
            is_test_a_pool = self.is_test_p
        else:
            is_test_a_pool = tf.constant(True)
        
        for i in range( self.n_lay_a_pool ):
            assert ( len(tdnn_sizes_after_pooling[i])==4 )
                
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_after_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_after_pooling[i][2]
            step_size = tdnn_sizes_after_pooling[i][3]
                                          
            self.layers_after_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                            n_step=n_step,
                                                                            in_dim=( para['W_1'].shape[0] - self.side_info_size[i + self.n_lay_b_pool] ) // n_step,
                                                                            out_dim=para['W_1'].shape[1], step_size=step_size,
                                                                            floatX=self.floatX, side_info_size=self.side_info_size[i+self.n_lay_b_pool]) )
            # Append the non-linearity
            self.layers_after_pooling.append(activations_after_pooling[i])
            
            # Append Batch-norm layer
            self.layers_after_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                           var=np.ones_like(para['b_1']),
                                                                                           offset=np.zeros_like(para['b_1']),
                                                                                           scale=np.ones_like(para['b_1']),
                                                                                           is_test =is_test_a_pool, decay=bn_decay,
                                                                                           floatX=self.floatX ) )    

        #########################################################################
        #### Multiclass classification.
        params_multi_class = tensorflow_code.initializers.init_params_simple_he_uniform( [tdnn_sizes_after_pooling[-1][1], self.n_spk],
                                                                                         floatX=self.floatX, use_bug=self.use_bug) 
        self.nn_multi_class = tensorflow_code.nn_def.tf_ff_nn(self.session, params_multi_class, floatX=self.floatX )
        
                                              
    def __call__(self,X1_, C1_, S1_, annoying_train):

        # Will only return what can be returned based on what is initialized.
        # For example, if layer_after_pooling is not initialized, embeddings
        # will not be returned but stats after pooling will. This is useful
        # when we will initialized the "after pooling norm" since the layers
        # afterwards have not been initialized at that stage.

        
        if len(self.residual_connections_bp) != 0:
            log.info("Residual connections " + str( self.residual_connections_bp ) )
            outputs_to_keep = np.unique( [i for l in self.residual_connections_bp for i in l ] )
            kept_outputs = {}


        
        ##########################################################################
        ### Apply pooling and the layers before it.
        #
        # With annoying_train all utterances are processed in on go before pooling
        # which allows proper batch norm. In testing, we procss them one by one.
        # Training
        if (annoying_train):
            def train_pooling():
                Z_ =  X1_
                """
                for j in range(0, len( self.layers_before_pooling ) ):
                    if isinstance( self.layers_before_pooling[j], tensorflow_code.nn_def.tf_tdnn ):
                        Z_ = self.layers_before_pooling[j]( Z_, S1_ )
                    else:
                        Z_ = self.layers_before_pooling[j]( Z_)
                        
                Y_train_ = tensorflow_code.pool_fkns.mean_std(Z_, axes=1)
                return Y_train_
                """

                for j in range(0, len( self.layers_before_pooling ) ):
                    if ( len(self.residual_connections_bp) > 0):

                        log.debug("Layer " + str(j) + " " + str(self.layers_before_pooling[j]) )
                        log.debug("Residuals  " + str(self.residual_connections_bp[j]) )
                        for o in self.residual_connections_bp[j]:
                            log.debug("   Adding " +str(o) + " " + str( kept_outputs[ o ] ) )

                            sh11_ = tf.shape(kept_outputs[ o ])[1]
                            sh21_ = tf.shape(Z_ )[1]
                            d1_   = tf.cast((sh11_ - sh21_)/2,"int32")
                            Z_r_ = kept_outputs[ o ][:, d1_:sh11_ - d1_,: ]
                            Z_ += Z_r_    #kept_outputs[ o ][d_:-d_]


                    if isinstance( self.layers_before_pooling[j], tensorflow_code.nn_def.tf_tdnn ):
                        Z_ = self.layers_before_pooling[j]( Z_, S1_ )
                    else:
                        Z_ = self.layers_before_pooling[j]( Z_)

                    if j in outputs_to_keep:
                        log.debug( "Layer " + str(j) + " " + str(self.layers_before_pooling[j]) + " will be stored. " )
                        kept_outputs[j] = Z_

                Y_train_ = tensorflow_code.pool_fkns.mean_std(Z_, axes=1)
                return Y_train_


            
            def test_pooling():
                nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling, pool_fkn =self.pool_function,
                                                                    output_size=self.pool_size, floatX=self.floatX)
                Y_test_ = nn_pool( X1_, C1_, S1_, res_con=self.residual_connections_bp ) 
                return Y_test_

            Y_ = tf.cond(self.is_test_p, test_pooling, train_pooling)

        # Without annoying_train we process the utterances one by one. This can reduces
        # memory requirements (by copying temporary intermediate results to the CPU RAM),
        # But with this option batch norm will not (yet) work properly in training. Should
        # fixed of-course if batch norm is shown to be useful.
        else:
            nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling, pool_fkn =self.pool_function,
                                                                output_size=self.pool_size, floatX=self.floatX )
            Y_ = nn_pool( X1_, C1_, S1_, res_con=self.residual_connections_bp )       
                                              
        # This variable is for stats in case we want to extract it.                                  
        stat_  = tf.squeeze(Y_, axis=[1], name='stats' )

        ##########################################################################                                              
        ### Layers after pooling
        embds_=[]
        """
        res_ap_ = []
        for i in range( len(self.layers_after_pooling) ):
            if isinstance( self.layers_after_pooling[i], tensorflow_code.nn_def.tf_tdnn ):
                Y_ = self.layers_after_pooling[i]( Y_, S1_ )
                # Extract embeddings after the last and the and second last TDNN
                embds_.append(Y_)
            else:
                Y_ = self.layers_after_pooling[i]( Y_ )
        """

###

        res_ap_ = []
        for i in range( len(self.layers_after_pooling) ):
            print (i)
            print ( self.layers_after_pooling[i] )
            if (i == self.stop_grad_ap ):
                log.info("Stopping gradient between between" + str(Y_))
                if isinstance( self.layers_after_pooling[i], tensorflow_code.nn_def.tf_tdnn ):
                    Y_ = self.layers_after_pooling[i]( tf.stop_gradient(Y_), S1_ ) # Gradiends will be stopped between layer i and i-1
                    embds_.append(Y_)
                else:
                    Y_ = self.layers_after_pooling[i]( tf.stop_gradient(Y_) ) # Gradiends will be stopped between layer i and i-1                    
                log.info("and " + str(Y_))
            else:
                if isinstance( self.layers_after_pooling[i], tensorflow_code.nn_def.tf_tdnn ):
                    Y_ = self.layers_after_pooling[i]( Y_, S1_ )
                    embds_.append(Y_)                                    
                else:
                    Y_ = self.layers_after_pooling[i]( Y_ )
                
                
            if isinstance(self.layers_after_pooling[i], tensorflow_code.nn_def.tf_batch_norm_mov_avg):
                res_ap_.append(Y_)

        # We don't want this to happen if layers after pooling have not been initialized.        
        if (len(res_ap_) > 0):
            for i in self.residual_ap:
                log.debug("Adding residual to softmax: " + str(res_ap_[i]) )
                Y_ += res_ap_[i]
        

###


        if (len(embds_) >= 2 ):
            embd_A_ = tf.squeeze(embds_[-2], axis=[1], name='embd_A' )
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        elif (len(embds_) >= 1 ):
            embd_A_ = None
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        else:
            embd_A_ = None
            embd_B_ = None 
            
        ##########################################################################
        ### Predictions
        if hasattr(self, 'nn_multi_class'):
            pred_ = tf.identity(self.nn_multi_class(tf.squeeze( Y_)), name='pred') # The identity is just to add the name.
        else:
            pred_ = None
        return stat_, embd_A_, embd_B_, pred_ 


    def get_parameters(self):
        params_ = []
        #if self.upd_b_pool:
        for l in self.layers_before_pooling:          #+ self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):       # Need to check this since e.g.relu doesn't have params
                params_ += l.get_parameters()  

        #if self.upd_a_pool:
        for l in self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):
                params_ += l.get_parameters() 
                    
        #if self.upd_multi:
        params_ +=  self.nn_multi_class.get_parameters() 
                    
        return params_

    
    def get_upd_parameters(self):
        params_ = []
        if self.upd_b_pool:
            #for l in self.layers_before_pooling  :
            #    if (hasattr(l, 'get_upd_parameters')):
            for i,l in enumerate(self.layers_before_pooling):
                if (hasattr(l, 'get_upd_parameters')) and self.layers_before_pooling_upd[i]:
                    params_ += l.get_upd_parameters() 

        if self.upd_a_pool:
            for l in self.layers_after_pooling:
                if (hasattr(l, 'get_upd_parameters')):
                    params_ += l.get_upd_parameters() 
                    
        if self.upd_multi:
            params_ +=  self.nn_multi_class.get_upd_parameters() 

        return params_

    def get_parameters_no_tdnn_side_info(self):
        params_ = []
        #if self.upd_b_pool:
        for l in self.layers_before_pooling:          #+ self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):       # Need to check this since e.g.relu doesn't have params
                if ( isinstance( l, tensorflow_code.nn_def.tf_tdnn ) and (len(l.get_parameters()) == 3) ):
                    params_ += l.get_parameters()[0:1]
                    params_ += l.get_parameters()[2:3]  
                else:                       
                    params_ += l.get_parameters()  

        #if self.upd_a_pool:
        for l in self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):
                if ( isinstance( l, tensorflow_code.nn_def.tf_tdnn ) and (len(l.get_parameters()) == 3) ):
                    params_ += l.get_parameters()[0:1]
                    params_ += l.get_parameters()[2:3]  
                else:                       
                    params_ += l.get_parameters()  
                    
        #if self.upd_multi:
        params_ +=  self.nn_multi_class.get_parameters() 
                    
        return params_

    
    def get_l2_reg(self, b_pool_reg=0.0, a_pool_reg=0.0, multi_reg=0.0):
        l2_reg = 0.0
        if (b_pool_reg > 0.0):
            for l in self.layers_before_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( b_pool_reg )
        if (a_pool_reg > 0.0):
            for l in self.layers_after_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( a_pool_reg )
        if (multi_reg > 0.0):
                    l2_reg += self.nn_multi_class.get_l2_reg( multi_reg )

        return l2_reg
###

class xvector_att(object):

    def __init__(self, session, is_test_p, n_spk, tdnn_sizes_before_pooling=None, tdnn_sizes_after_pooling=None,
                 activations_before_pooling=None, activations_after_pooling=None, pool_function=None, pool_size=None,
                 it_tr_que=None, scp_info=None, load_data_function=None, upd_b_pool=True, upd_att=True, upd_a_pool=True,
                 upd_multi=True, do_feat_norm=False, upd_feat_norm=False, do_pool_norm=False, upd_pool_norm=False,
                 att_n_heads=3, att_dim=10, att_activation=tf.nn.relu, use_bug=False, floatX='float32'):

        # Attention parameters
        self.att_n_heads    = att_n_heads
        self.att_dim        = att_dim
        self.att_activation = att_activation
        
        # Perhaps not great to have it here, but layers require it for their get_parameters functions etc.
        # Alternative would be to always pass sessions to these calls or use default session. ... Think about this.
        # Same with the is_test_p variable, Could have been passed with each call to the model but not sure if this
        # is more convenient.
        self.session = session
        self.is_test_p = is_test_p

        # Whether to use the buggy initialization
        self.use_bug=use_bug
        
        # If no architecture is given, we assume the JHU one.
        if tdnn_sizes_before_pooling == None:
            self.tdnn_sizes_before_pooling = ( [115, 512, 5, 1], [1536, 512, 3, 2], [1536,512, 3, 3],
                                                 [512, 512, 1,1], [512,1500,1,1] )
        else:
            self.tdnn_sizes_before_pooling = tdnn_sizes_before_pooling
       
        if tdnn_sizes_after_pooling == None:
            self.tdnn_sizes_after_pooling = ( [3000, 512,1,1], [512, 512,1,1] ) # Actually, these would normally be normal dense layers
        else:                                                                    
            self.tdnn_sizes_after_pooling = tdnn_sizes_after_pooling           

        self.n_lay_b_pool   = len(tdnn_sizes_before_pooling)
        self.feat_dim       = self.tdnn_sizes_before_pooling[0][0] / self.tdnn_sizes_before_pooling[0][2] # Input_dim / n_step
        self.n_lay_a_pool   = len(tdnn_sizes_after_pooling)

        
        if activations_before_pooling == None:        
            self.activations_before_pooling = [tf.nn.relu] * self.n_lay_b_pool
        else:
            self.activations_before_pooling = activations_before_pooling

        if activations_after_pooling  == None:        
            self.activations_after_pooling = [tf.nn.relu] * self.n_lay_a_pool
        else:
            self.activations_after_pooling = activations_after_pooling            

        if pool_function == None:
            # self.pool_function = lambda x: tensorflow_code.pool_fkns.mean_std(x, axes=1)
            # pool_function = lambda x, x_org: tensorflow_code.pool_fkns.mean_std_attention(x, tf.tile( tf.nn.softmax(tdnn_attention(x_org)), [1,1,10]), axes=1)
            # self.pool_function = lambda x_, a_: tensorflow_code.pool_fkns.mean_std_attention(x_, a_), axes=1)
            self.pool_function = lambda x_: tensorflow_code.pool_fkns.mean_std_attention_head(x_, self.att_layer(x_), axes=1) 
        else:
            self.pool_function = pool_function

        if pool_size == None:
            self.pool_size = self.tdnn_sizes_before_pooling[-1][1] * 2 * self.att_n_heads
        else:    
            self.pool_size = pool_size 

        log.info("TDNN architecture before pooling: " + str(list(zip(self.tdnn_sizes_before_pooling,self.activations_before_pooling))) )
        log.info("TDNN architecture after pooling: "  + str(list(zip(self.tdnn_sizes_after_pooling,self.activations_after_pooling))) )       
        
        self.floatX        = floatX
        self.is_test_p     = is_test_p
        self.upd_b_pool    = upd_b_pool
        self.upd_a_pool    = upd_a_pool
        self.upd_multi     = upd_multi
        self.do_feat_norm  = do_feat_norm
        self.upd_feat_norm = upd_feat_norm
        self.do_pool_norm  = do_pool_norm
        self.upd_pool_norm = upd_pool_norm
        self.upd_att       = upd_att
        #self.it_tr         = it_tr
        self.n_spk         = n_spk
        
        log.info('Initializing model randomly')
        np.random.seed(17)
                
        self.n_lay_b_pool = len(tdnn_sizes_before_pooling)
        self.n_lay_a_pool = len(tdnn_sizes_after_pooling)

        bn_decay = 0.95 # Decay rate for batch-norm.

        self.layers_before_pooling = []            
        self.layers_after_pooling  = []                                               

        
        ##############################################################################
        ### Estimate normalization of feats if desired.        
        if (do_feat_norm):
            
            if (it_tr_que != None ):
                log.info("Estimating feature normalization")
                [X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()

                ### Apply the normalization    
                mean_feat = np.mean(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
                var_feat  = np.var(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
            else:
                log.info("Feature normalization initialized with mean=0, std=1" )                
                mean_feat = np.zeros([1,1,self.feat_dim])
                var_feat  = np.ones([1,1,self.feat_dim])
                
            if ( self.upd_feat_norm ):
                log.info("Feature norm will be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
            else:
                log.info("Feature norm will not be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
            self.layers_before_pooling +=  [bn_feats] 
    
        ##########################################################################        
        ### Layers before pooling
        if self.upd_b_pool:
            is_test_b_pool = self.is_test_p
        else:
            is_test_b_pool = tf.constant(True)

        for i in range( self.n_lay_b_pool ):
            assert ( len(tdnn_sizes_before_pooling[i])==4 )
                
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_before_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_before_pooling[i][2]
            step_size = tdnn_sizes_before_pooling[i][3]

            self.layers_before_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                             n_step=n_step,
                                                                             in_dim=para['W_1'].shape[0] / n_step,
                                                                             out_dim=para['W_1'].shape[1],
                                                                             step_size=step_size,
                                                                             floatX=self.floatX) )

            # Append the non-linearity
            self.layers_before_pooling.append(activations_before_pooling[i])
            
            # Append Batch-norm layer
            self.layers_before_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                            var=np.ones_like(para['b_1']),
                                                                                            offset=np.zeros_like(para['b_1']),
                                                                                            scale=np.ones_like(para['b_1']),
                                                                                            is_test =is_test_b_pool, decay=bn_decay,
                                                                                            floatX=self.floatX ) )
            
        ##########################################################################        
        ### Attention
        para_1 = tensorflow_code.initializers.init_params_simple_he_uniform( (self.tdnn_sizes_before_pooling[-1][1], self.att_dim),
                                                                             floatX=self.floatX, use_bug=self.use_bug)
        para_2 = tensorflow_code.initializers.init_params_simple_he_uniform( (self.att_dim, self.att_n_heads), floatX=self.floatX, use_bug=self.use_bug)

        n_step=1
        step_size=1
        self.att_layer = tensorflow_code.nn_def.tf_self_att_simple(self.session, in_dim=para_1['W_1'].shape[0] / n_step, out_dim=para_1['W_1'].shape[1],
                                                                   weight_1 =para_1['W_1'], weight_2=para_2['W_1'], bias_1=para_1['b_1'], bias_2=para_2['b_1'],
                                                                   activation = self.att_activation, n_step=n_step, step_size=step_size, floatX=self.floatX)

        ##########################################################################
        ### Add and estimate normalization of output from pooling layer if desired.
        #
        # This is the only "architecture difference" from the Kaldi. In order to
        # reach the same performance as Kaldi, we have to make sure that the
        # output from the pooling layer has mean 0 and standard deviation 1. We 
        # achieve this by estimating the output mean and standard deviation on a
        # few batches here after initializing the model and then use this statatistics
        # for normalization. Most likely, Kaldi's optimizer is more robust to this.
           
        # Placeholder that will be used in initialization
        if (do_pool_norm):

            if (it_tr_que != None ):
                log.info("Estimating statistics of pool output")
                X1_p        = tf.placeholder(self.floatX, shape=[None,None,self.feat_dim], name='X1_p') # Features 
                C1_p        = tf.placeholder(dtype='int32', shape=[None], name='C1_p')                  # Indices
                Y_, _, _, _ = self.__call__(X1_p, C1_p,annoying_train=True)                             # The output of the pooling layer.
                g_stat      = lambda X1 : self.session.run(Y_, {X1_p: X1, C1_p: np.array([]), self.is_test_p:False})
                ss          = np.zeros([0,self.pool_size])
                print(ss.shape)
                
                self.session.run(tf.global_variables_initializer()) 
                log.info("Calculating mean and standard deviation of pooling output") 
                for i in range(10):
                    [X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                    print(g_stat(feats).squeeze().shape)
                    ss    = np.concatenate([ss,g_stat(feats).squeeze()], axis=0)
                
                ### Apply the normalization as a batchnorm layer   
                mean_pool = np.mean(ss,axis=0)[np.newaxis,np.newaxis,:]
                var_pool  = np.var(ss,axis=0)[np.newaxis,np.newaxis,:]

            else:
                log.info("Pool normalization initialized with mean=0, std=1")
                mean_pool = np.zeros([1,1,self.pool_size])
                var_pool  = np.ones([1,1,self.pool_size])
                
            if ( self.upd_pool_norm ):
                log.info("Pool norm will be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX ) 
            else:
                log.info("Pool norm will not be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX )              
            self.layers_after_pooling.append(bn_pool)
        ##########################################################################        
        ### Layers after pooling
        if self.upd_a_pool:
            is_test_a_pool = self.is_test_p
        else:
            is_test_a_pool = tf.constant(True)

        for i in range( self.n_lay_a_pool ):
            assert ( len(tdnn_sizes_after_pooling[i])==4 )
                
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_after_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_after_pooling[i][2]
            step_size = tdnn_sizes_after_pooling[i][3]
                                          
            self.layers_after_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                            n_step=n_step,
                                                                            in_dim=para['W_1'].shape[0] / n_step,
                                                                            out_dim=para['W_1'].shape[1],
                                                                            step_size=step_size, floatX=self.floatX) )
            # Append the non-linearity
            self.layers_after_pooling.append(activations_after_pooling[i])
            
            # Append Batch-norm layer
            self.layers_after_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                           var=np.ones_like(para['b_1']),
                                                                                           offset=np.zeros_like(para['b_1']),
                                                                                           scale=np.ones_like(para['b_1']),
                                                                                           is_test =is_test_a_pool, decay=bn_decay,
                                                                                           floatX=self.floatX ) )    

        #########################################################################
        #### Multiclass classification.
        params_multi_class = tensorflow_code.initializers.init_params_simple_he_uniform( [tdnn_sizes_after_pooling[-1][1], self.n_spk],
                                                                                         floatX=self.floatX, use_bug=self.use_bug) 
        self.nn_multi_class = tensorflow_code.nn_def.tf_ff_nn(self.session, params_multi_class, floatX=self.floatX )
        
                                              
    def __call__(self,X1_, C1_, annoying_train):

        # Will only return what can be returned based on what is initialized.
        # For example, if layer_after_pooling is not initialized, embeddings
        # will not be returned but stats after pooling will. This is useful
        # when we will initialized the "after pooling norm" since the layers
        # afterwards have not been initialized at that stage.
        
        ##########################################################################
        ### Apply pooling and the layers before it.
        #
        # With annoying_train all utterances are processed in on go before pooling
        # which allows proper batch norm. In testing, we procss them one by one.
        # Training
        if (annoying_train):
            def train_pooling():
                Z_ =  X1_
                for j in range(0, len( self.layers_before_pooling ) ):
                    Z_ = self.layers_before_pooling[j]( Z_ )

                Y_train_ = self.pool_function(Z_)
                return Y_train_

            def test_pooling():
                nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling, pool_fkn =self.pool_function,
                                                                output_size=self.pool_size, floatX=self.floatX )
                Y_test_ = nn_pool( X1_, C1_ )
                return Y_test_

            Y_ = tf.cond(self.is_test_p, test_pooling, train_pooling)

        # Without annoying_train we process the utterances one by one. This can reduces
        # memory requirements (by copying temporary intermediate results to the CPU RAM),
        # But with this option batch norm will not (yet) work properly in training. Should
        # fixed of-course if batch norm is shown to be useful.
        else:
            nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling , pool_fkn =self.pool_function,
                                                            output_size=self.pool_size, floatX=self.floatX )
            Y_ = nn_pool( X1_, C1_ )       
                                              
        # This variable is for stats in case we want to extract it.                                  
        stat_  = tf.squeeze(Y_, axis=[1], name='stats' )

        
        

        ##########################################################################                                              
        ### Layers after pooling
        embds_=[]
        for i in range( len(self.layers_after_pooling) ):
            Y_ = self.layers_after_pooling[i]( Y_ )
                                              
            # Extract embeddings after the last and the and second last TDNN
            if isinstance(self.layers_after_pooling[i], tensorflow_code.nn_def.tf_tdnn):
                embds_.append(Y_)

        if (len(embds_) >= 2 ):
            embd_A_ = tf.squeeze(embds_[-2], axis=[1], name='embd_A' )
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        elif (len(embds_) >= 1 ):
            embd_A_ = None
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        else:
            embd_A_ = None
            embd_B_ = None 
            
        ##########################################################################
        ### Predictions
        if hasattr(self, 'nn_multi_class'):
            pred_ = tf.identity(self.nn_multi_class(tf.squeeze( Y_)), name='pred') # The identity is just to add the name.
        else:
            pred_ = None
        return stat_, embd_A_, embd_B_, pred_ 


    def get_parameters(self):
        params_ = []
        #if self.upd_b_pool:
        for l in self.layers_before_pooling:          #+ self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):       # Need to check this since e.g.relu doesn't have params
                params_ += l.get_parameters()

        #if self.upd_att:
        params_ += self.att_layer.get_parameters()

        #if self.upd_a_pool:
        for l in self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):
                params_ += l.get_parameters() 
                    
        #if self.upd_multi:
        params_ +=  self.nn_multi_class.get_parameters() 

                    
        return params_
       
    def get_upd_parameters(self):
        params_ = []
        if self.upd_b_pool:
            for l in self.layers_before_pooling  :
                if (hasattr(l, 'get_upd_parameters')):
                    params_ += l.get_upd_parameters()

        if self.upd_att:
            params_ += self.att_layer.get_upd_parameters()

        if self.upd_a_pool:
            for l in self.layers_after_pooling:
                if (hasattr(l, 'get_upd_parameters')):
                    params_ += l.get_upd_parameters() 
                    
        if self.upd_multi:
            params_ +=  self.nn_multi_class.get_upd_parameters() 

        return params_

    
    def get_l2_reg(self, b_pool_reg=0.0, att_reg=0.0, a_pool_reg=0.0, multi_reg=0.0):
        l2_reg = 0.0
        if (b_pool_reg > 0.0):
            for l in self.layers_before_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( b_pool_reg )
        if (att_reg > 0.0):
            l2_reg += self.att_layer.get_l2_reg( att_reg )
        if (a_pool_reg > 0.0):
            for l in self.layers_after_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( a_pool_reg )
        if (multi_reg > 0.0):
                    l2_reg += self.nn_multi_class.get_l2_reg( multi_reg )

        return l2_reg


    
class dplda_plus(object):

    def __init__(self, session, is_test_p, embd_, in_size, red_size, floatX='float32',
                 unit_l_norm=True, use_b_norm=True, use_b_norm_2=False, do_l_norm=True):

        from tensorflow_code.dplda import dplda_model
    
        self.session   = session
        self.is_test_p = is_test_p
        self.red_size  = red_size
        self.unit_l_norm = unit_l_norm
        self.use_b_norm_2 = use_b_norm_2
        self.do_l_norm = do_l_norm

        
        self.mean_       = tf.Variable(np.zeros(in_size), name='dplda_input_mean', dtype=floatX )
        lda_tmp          = tensorflow_code.initializers.init_params_simple_he_uniform( (in_size, red_size), floatX=floatX)
        self.lda_        = tf.Variable( lda_tmp['W_1'].T, name='dplda_lda', dtype=floatX) 
        self.lda_offset_ = tf.Variable( lda_tmp['b_1'], name='dplda_lda_offset', dtype=floatX)

        if use_b_norm:
            self.bn = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros(in_size),
                                                                   var=np.ones(in_size),
                                                                   offset=np.zeros(in_size),
                                                                   scale=np.ones(in_size),
                                                                   is_test =self.is_test_p, decay=0.95,
                                                                   floatX=floatX ) 
            embd_ = tf.squeeze(self.bn(tf.expand_dims(embd_,0) ))
        else:
            self.bn = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros(in_size),
                                                                   var=np.ones(in_size),
                                                                   offset=np.zeros(in_size),
                                                                   scale=np.ones(in_size),
                                                                   is_test =tf.constant(True), decay=0.95,
                                                                   floatX=floatX ) 
            embd_ = tf.squeeze(self.bn(tf.expand_dims(embd_,0) ))

            
        m = np.zeros(red_size).astype(floatX)
        B = np.diag(np.ones(red_size).astype(floatX))
        W = np.diag(np.ones(red_size).astype(floatX) )

        if use_b_norm_2:
            self.bn2 = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros( self.red_size ),
                                                                    var=np.ones( self.red_size ),
                                                                    offset=np.zeros( self.red_size ),
                                                                    scale=np.ones( self.red_size ),
                                                                    is_test =self.is_test_p, decay=0.95,
                                                                    floatX=floatX ) 


            self.dplda=dplda_model(session, [m,B,W], tf.squeeze( self.bn2( tf.expand_dims( self.kaldi_pre_proc(embd_), 0) ) ), floatX=floatX)
        else:
            self.dplda=dplda_model(session, [m,B,W], self.kaldi_pre_proc(embd_), floatX=floatX)
        

    def kaldi_pre_proc(self, X_):
        Z_ = X_ - self.mean_                                                        # subtract global mean
        Z_ = tf.matmul(Z_, self.lda_, transpose_b=True) - self.lda_offset_          # LDA
        if self.do_l_norm:
            Z_ = Z_ / tf.norm(Z_, axis=1, keep_dims=True) #* np.sqrt( self.red_size )   # l-norm
            if not self.unit_l_norm:
                Z_ *= np.sqrt( self.red_size )                                          # Kaldi style. 
        #Z_ = tf.matmul( (Z_ - m), plda_tr, transpose_b=True)                 # Sim. diag Skip this since we use full matrices in DPLDA
        return Z_

    def __call__():
        pass

    def make_loss(self, loss_fcn, W, Y, tau, l2_reg_P, l2_reg_Q, l2_reg_c, l2_reg_k,
                  reg_to=[], l2_reg_score=0.0, sep_reg=False, batch_score_norm=False):
        return self.dplda.make_loss(loss_fcn, W, Y, tau, l2_reg_P, l2_reg_Q, l2_reg_c, l2_reg_k,
                                    reg_to, l2_reg_score, sep_reg, batch_score_norm)
        
    def get_parameters(self):
        if self.use_b_norm_2:
            return self.bn.get_parameters() + [self.mean_, self.lda_, self.lda_offset_] + self.dplda.get_parameters() + self.bn2.get_parameters()
        else:
            return self.bn.get_parameters() + [self.mean_, self.lda_, self.lda_offset_] + self.dplda.get_parameters()
        
    def get_upd_parameters(self):
        if self.use_b_norm_2:
            return self.bn.get_upd_parameters() + [self.mean_, self.lda_, self.lda_offset_] + self.dplda.get_parameters() + self.bn2.get_upd_parameters()
        else:
            return self.bn.get_upd_parameters() + [self.mean_, self.lda_, self.lda_offset_] + self.dplda.get_parameters() 




class xvector_side_info(object):

    def __init__(self, session, is_test_p, n_spk, tdnn_sizes_before_pooling=None, tdnn_sizes_after_pooling=None,
                 activations_before_pooling=None, activations_after_pooling=None, pool_function=None, pool_size=None,
                 it_tr_que=None, scp_info=None, load_data_function=None, upd_b_pool=True, upd_a_pool=True,
                 upd_multi=True, do_feat_norm=False, upd_feat_norm=False, do_pool_norm=False, upd_pool_norm=False,
                 side_info_sizes=[], upd_b_pool_spec=[], use_bug=False, floatX='float32'):

        # Perhaps not great to have it here, but layers require it for their get_parameters functions etc.
        # Alternative would be to always pass sessions to these calls or use default session. ... Think about this.
        # Same with the is_test_p variable, Could have been passed with each call to the model but not sure if this
        # is more convenient.
        self.session = session
        self.is_test_p = is_test_p

        # Whether to use the buggy initialization
        self.use_bug=use_bug

        # The size of the side input to each tdnn layer
        self.side_info_size = side_info_sizes
        if len(self.side_info_size) == 0:
            self.side_info_size = [0] * (len(tdnn_sizes_before_pooling) + len(tdnn_sizes_after_pooling))
        assert(len(self.side_info_size) == (len(tdnn_sizes_before_pooling) + len(tdnn_sizes_after_pooling)) )

        # If no architecture is given, we assume the JHU one.
        if tdnn_sizes_before_pooling == None:
            self.tdnn_sizes_before_pooling = ( [115, 512, 5, 1], [1536, 512, 3, 2], [1536,512, 3, 3],
                                                 [512, 512, 1,1], [512,1500,1,1] )
        else:
            self.tdnn_sizes_before_pooling = tdnn_sizes_before_pooling
    
        if tdnn_sizes_after_pooling == None:
            self.tdnn_sizes_after_pooling = ( [3000, 512,1,1], [512, 512,1,1] ) # Actually, these would normally be normal dense layers
        else:                                                                    
            self.tdnn_sizes_after_pooling = tdnn_sizes_after_pooling           

        
        self.n_lay_b_pool   = len(tdnn_sizes_before_pooling)
        self.feat_dim       = self.tdnn_sizes_before_pooling[0][0] / self.tdnn_sizes_before_pooling[0][2] # Input_dim / n_step
        self.n_lay_a_pool   = len(tdnn_sizes_after_pooling)

    
        # Adjust the tdnn size to include th side info
        for i in range( self.n_lay_b_pool ):
            self.tdnn_sizes_before_pooling[i][0] += self.side_info_size[i]
        for i in range( self.n_lay_a_pool ):
            self.tdnn_sizes_after_pooling[i][0] += self.side_info_size[ i + self.n_lay_b_pool ]
            
        if activations_before_pooling == None:        
            self.activations_before_pooling = [tf.nn.relu] * self.n_lay_b_pool
        else:
            self.activations_before_pooling = activations_before_pooling

        if activations_after_pooling  == None:        
            self.activations_after_pooling = [tf.nn.relu] * self.n_lay_a_pool
        else:
            self.activations_after_pooling = activations_after_pooling            

        if pool_function == None:
            self.pool_function = lambda x: tensorflow_code.pool_fkns.mean_std(x, axes=1)
        else:
            self.pool_function = pool_function

        if pool_size == None:
            self.pool_size = self.tdnn_sizes_before_pooling[-1][1] * 2
        else:    
            self.pool_size = pool_size 

        log.info("TDNN architecture before pooling: " + str(list(zip(self.tdnn_sizes_before_pooling,self.activations_before_pooling))) )
        log.info("TDNN architecture after pooling: "  + str(list(zip(self.tdnn_sizes_after_pooling,self.activations_after_pooling))) )       
        
        self.floatX        = floatX
        self.is_test_p     = is_test_p
        self.upd_b_pool    = upd_b_pool
        self.upd_a_pool    = upd_a_pool
        self.upd_multi     = upd_multi
        self.do_feat_norm  = do_feat_norm
        self.upd_feat_norm = upd_feat_norm
        self.do_pool_norm  = do_pool_norm
        self.upd_pool_norm = upd_pool_norm
        #self.it_tr         = it_tr
        self.n_spk         = n_spk

        self.upd_b_pool_spec = upd_b_pool_spec

        assert( (len(self.upd_b_pool_spec) == 0) or  (len(self.upd_b_pool_spec) == self.n_lay_b_pool) ) 
        if (len(self.upd_b_pool_spec) == 0):
            self.upd_b_pool_spec = [True] * self.n_lay_b_pool
            if (not self.upd_b_pool):
                log.warning("Providing upd_b_pool_spec is meaningless if upd_b_pool=False")
                self.upd_b_pool_spec = [False] * self.n_lay_b_pool # Just to be extra sure

        
        log.info('Initializing model randomly')
        np.random.seed(17)
        #lda_tmp    = tensorflow_code.initializers.init_params_simple_he_uniform( (512, 150),
        #                                                                         floatX=floatX, use_bug=self.use_bug) # Delete this

                
        self.n_lay_b_pool = len(tdnn_sizes_before_pooling)
        self.n_lay_a_pool = len(tdnn_sizes_after_pooling)

        bn_decay = 0.95 # Decay rate for batch-norm.

        self.layers_before_pooling = []            
        self.layers_after_pooling  = []                                               
        self.layers_before_pooling_upd = []
        ##############################################################################
        ### Estimate normalization of feats if desired.        
        if (do_feat_norm):
            
            if (it_tr_que != None ):
                log.info("Estimating feature normalization")
                [X, Y, U, S], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()

                ### Apply the normalization    
                mean_feat = np.mean(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
                var_feat  = np.var(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
            else:
                log.info("Feature normalization initialized with mean=0, std=1" )                
                mean_feat = np.zeros([1,1,self.feat_dim])
                var_feat  = np.ones([1,1,self.feat_dim])
                
            if ( self.upd_feat_norm ):
                log.info("Feature norm will be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
                self.layers_before_pooling_upd +=  [ True ] 
            else:
                log.info("Feature norm will not be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
                self.layers_before_pooling_upd +=  [ False ] 
            self.layers_before_pooling +=  [bn_feats] 
    
        ##########################################################################        
        ### Layers before pooling

        for i in range( self.n_lay_b_pool ):
            assert ( len(tdnn_sizes_before_pooling[i])==4 )

            if self.upd_b_pool and self.upd_b_pool_spec[i]:
                is_test_b_pool = self.is_test_p
            else:
                is_test_b_pool = tf.constant(True)
            
                
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_before_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_before_pooling[i][2]
            step_size = tdnn_sizes_before_pooling[i][3]

            self.layers_before_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                             n_step=n_step,
                                                                             in_dim=( para['W_1'].shape[0] - self.side_info_size[i] ) // n_step, 
                                                                             out_dim=para['W_1'].shape[1],
                                                                             step_size=step_size,
                                                                             floatX=self.floatX, side_info_size=self.side_info_size[i]) )


            # Append the non-linearity
            self.layers_before_pooling.append(activations_before_pooling[i])
            
            # Append Batch-norm layer
            self.layers_before_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                            var=np.ones_like(para['b_1']),
                                                                                            offset=np.zeros_like(para['b_1']),
                                                                                            scale=np.ones_like(para['b_1']),
                                                                                            is_test =is_test_b_pool, decay=bn_decay,
                                                                                            floatX=self.floatX ) )
            if self.upd_b_pool and self.upd_b_pool_spec[i]:
                self.layers_before_pooling_upd +=  [ True, True, True ]
            else:
                self.layers_before_pooling_upd +=  [ False, False, False ]
            
        ##########################################################################
        ### Add and estimate normalization of output from pooling layer if desired.
        #
        # This is the only "architecture difference" from the Kaldi. In order to
        # reach the same performance as Kaldi, we have to make sure that the
        # output from the pooling layer has mean 0 and standard deviation 1. We 
        # achieve this by estimating the output mean and standard deviation on a
        # few batches here after initializing the model and then use this statatistics
        # for normalization. Most likely, Kaldi's optimizer is more robust to this.
           
        # Placeholder that will be used in initialization
        if (do_pool_norm):

            if (it_tr_que != None ):
                log.info("Estimating statistics of pool output")
                X1_p        = tf.placeholder(self.floatX, shape=[None,None,self.feat_dim], name='X1_p') # Features 
                C1_p        = tf.placeholder(dtype='int32', shape=[None], name='C1_p')                  # Indices
                S1_p        = tf.placeholder(dtype='int32', shape=[None,1], name='S1_p')                  # Indices
                Y_, _, _, _ = self.__call__(X1_p, C1_p, S1_p, annoying_train=True)                             # The output of the pooling layer.
                g_stat      = lambda X1, S1 : self.session.run(Y_, {X1_p: X1, C1_p: np.array([]), S1_p: S1, self.is_test_p:False})
                ss          = np.zeros([0,self.pool_size])
                
                self.session.run(tf.global_variables_initializer()) 
                log.info("Calculating mean and standard deviation of pooling output") 
                for i in range(10):
                    #[X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                    [X, Y, U, S], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                    ss    = np.concatenate([ss,g_stat(feats, np.vstack(S)).squeeze()], axis=0)
                
                ### Apply the normalization as a batchnorm layer   
                mean_pool = np.mean(ss,axis=0)[np.newaxis,np.newaxis,:]
                var_pool  = np.var(ss,axis=0)[np.newaxis,np.newaxis,:]

            else:
                log.info("Pool normalization initialized with mean=0, std=1")
                mean_pool = np.zeros([1,1,self.pool_size])
                var_pool  = np.ones([1,1,self.pool_size])
                
            if ( self.upd_pool_norm ):
                log.info("Pool norm will be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX ) 
            else:
                log.info("Pool norm will not be updated")
                bn_pool   = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_pool, var=var_pool,
                                                                         offset=np.zeros_like( mean_pool ),
                                                                         scale=np.ones_like( var_pool ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.001, floatX=self.floatX )              
            self.layers_after_pooling.append(bn_pool)
        ##########################################################################        
        ### Layers after pooling
        if self.upd_a_pool:
            is_test_a_pool = self.is_test_p
        else:
            is_test_a_pool = tf.constant(True)
        
        for i in range( self.n_lay_a_pool ):
            assert ( len(tdnn_sizes_after_pooling[i])==4 )
                
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes_after_pooling[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes_after_pooling[i][2]
            step_size = tdnn_sizes_after_pooling[i][3]
                                          
            self.layers_after_pooling.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                            n_step=n_step,
                                                                            in_dim=( para['W_1'].shape[0] - self.side_info_size[i + self.n_lay_b_pool] ) // n_step,
                                                                            out_dim=para['W_1'].shape[1], step_size=step_size,
                                                                            floatX=self.floatX, side_info_size=self.side_info_size[i+self.n_lay_b_pool]) )
            # Append the non-linearity
            self.layers_after_pooling.append(activations_after_pooling[i])
            
            # Append Batch-norm layer
            self.layers_after_pooling.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                           var=np.ones_like(para['b_1']),
                                                                                           offset=np.zeros_like(para['b_1']),
                                                                                           scale=np.ones_like(para['b_1']),
                                                                                           is_test =is_test_a_pool, decay=bn_decay,
                                                                                           floatX=self.floatX ) )    

        #########################################################################
        #### Multiclass classification.
        params_multi_class = tensorflow_code.initializers.init_params_simple_he_uniform( [tdnn_sizes_after_pooling[-1][1], self.n_spk],
                                                                                         floatX=self.floatX, use_bug=self.use_bug) 
        self.nn_multi_class = tensorflow_code.nn_def.tf_ff_nn(self.session, params_multi_class, floatX=self.floatX )
        
                                              
    def __call__(self,X1_, C1_, S1_, annoying_train):

        # Will only return what can be returned based on what is initialized.
        # For example, if layer_after_pooling is not initialized, embeddings
        # will not be returned but stats after pooling will. This is useful
        # when we will initialized the "after pooling norm" since the layers
        # afterwards have not been initialized at that stage.
        
        ##########################################################################
        ### Apply pooling and the layers before it.
        #
        # With annoying_train all utterances are processed in on go before pooling
        # which allows proper batch norm. In testing, we procss them one by one.
        # Training
        if (annoying_train):
            def train_pooling():
                Z_ =  X1_           
                for j in range(0, len( self.layers_before_pooling ) ):
                    if isinstance( self.layers_before_pooling[j], tensorflow_code.nn_def.tf_tdnn ):
                        Z_ = self.layers_before_pooling[j]( Z_, S1_ )
                    else:
                        Z_ = self.layers_before_pooling[j]( Z_)
                        
                Y_train_ = tensorflow_code.pool_fkns.mean_std(Z_, axes=1)
                return Y_train_

            def test_pooling():
                nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling, pool_fkn =self.pool_function,
                                                                    output_size=self.pool_size, floatX=self.floatX)
                Y_test_ = nn_pool( X1_, C1_, S1_ ) 
                return Y_test_

            Y_ = tf.cond(self.is_test_p, test_pooling, train_pooling)

        # Without annoying_train we process the utterances one by one. This can reduces
        # memory requirements (by copying temporary intermediate results to the CPU RAM),
        # But with this option batch norm will not (yet) work properly in training. Should
        # fixed of-course if batch norm is shown to be useful.
        else:
            nn_pool     = tensorflow_code.nn_def.tf_pool( self.layers_before_pooling, pool_fkn =self.pool_function,
                                                                output_size=self.pool_size, floatX=self.floatX )
            Y_ = nn_pool( X1_, C1_, S1_ )       
                                              
        # This variable is for stats in case we want to extract it.                                  
        stat_  = tf.squeeze(Y_, axis=[1], name='stats' )

        ##########################################################################                                              
        ### Layers after pooling
        embds_=[]
        for i in range( len(self.layers_after_pooling) ):
            if isinstance( self.layers_after_pooling[i], tensorflow_code.nn_def.tf_tdnn ):
                Y_ = self.layers_after_pooling[i]( Y_, S1_ )
                # Extract embeddings after the last and the and second last TDNN
                embds_.append(Y_)
            else:
                Y_ = self.layers_after_pooling[i]( Y_ )
                                              

        if (len(embds_) >= 2 ):
            embd_A_ = tf.squeeze(embds_[-2], axis=[1], name='embd_A' )
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        elif (len(embds_) >= 1 ):
            embd_A_ = None
            embd_B_ = tf.squeeze(embds_[-1], axis=[1], name='embd_B' )
        else:
            embd_A_ = None
            embd_B_ = None 
            
        ##########################################################################
        ### Predictions
        if hasattr(self, 'nn_multi_class'):
            pred_ = tf.identity(self.nn_multi_class(tf.squeeze( Y_)), name='pred') # The identity is just to add the name.
        else:
            pred_ = None
        return stat_, embd_A_, embd_B_, pred_ 


    def get_parameters(self):
        params_ = []
        #if self.upd_b_pool:
        for l in self.layers_before_pooling:          #+ self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):       # Need to check this since e.g.relu doesn't have params
                params_ += l.get_parameters()  

        #if self.upd_a_pool:
        for l in self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):
                params_ += l.get_parameters() 
                    
        #if self.upd_multi:
        params_ +=  self.nn_multi_class.get_parameters() 

                    
        return params_
       
    def get_upd_parameters(self):
        params_ = []
        if self.upd_b_pool:
            #for l in self.layers_before_pooling  :
            #    if (hasattr(l, 'get_upd_parameters')):
            for i,l in enumerate(self.layers_before_pooling):
                if (hasattr(l, 'get_upd_parameters')) and self.layers_before_pooling_upd[i]:
                    params_ += l.get_upd_parameters() 

        if self.upd_a_pool:
            for l in self.layers_after_pooling:
                if (hasattr(l, 'get_upd_parameters')):
                    params_ += l.get_upd_parameters() 
                    
        if self.upd_multi:
            params_ +=  self.nn_multi_class.get_upd_parameters() 

        return params_

    
    def get_l2_reg(self, b_pool_reg=0.0, a_pool_reg=0.0, multi_reg=0.0):
        l2_reg = 0.0
        if (b_pool_reg > 0.0):
            for l in self.layers_before_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( b_pool_reg )
        if (a_pool_reg > 0.0):
            for l in self.layers_after_pooling:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( a_pool_reg )
        if (multi_reg > 0.0):
                    l2_reg += self.nn_multi_class.get_l2_reg( multi_reg )

        return l2_reg


#######################3
class tdnn_stack(object):

    def __init__(self, session, is_test_p, tdnn_sizes=None, activations=None, it_tr_que=None,
                 upd_tdnn=True, do_feat_norm=False, upd_feat_norm=False, side_info_sizes=[],
                 upd_tdnn_spec=[], apply_bn_norm=[], use_bug=False, floatX='float32'):

        # Perhaps not great to have it here, but layers require it for their get_parameters functions etc.
        # Alternative would be to always pass sessions to these calls or use default session. ... Think about this.
        # Same with the is_test_p variable, Could have been passed with each call to the model but not sure if this
        # is more convenient.
        self.session = session
        self.is_test_p = is_test_p

        # Whether to use the buggy initialization
        self.use_bug=use_bug
        
        # The size of the side input to each tdnn layer
        self.side_info_size = side_info_sizes
        if len(self.side_info_size) == 0:
            self.side_info_size = [0] * len(tdnn_sizes) 
        assert(len(self.side_info_size) == len(tdnn_sizes))

        # If no architecture is given, we assume the JHU before pooling one.
        if tdnn_sizes == None:
            self.tdnn_sizes = ( [115, 512, 5, 1], [1536, 512, 3, 2], [1536,512, 3, 3], [512, 512, 1,1], [512,1500,1,1] )
        else:
            self.tdnn_sizes = tdnn_sizes
        
        self.n_lay    = len( tdnn_sizes )
        self.feat_dim = self.tdnn_sizes [0][0] / self.tdnn_sizes[0][2] # Input_dim / n_step

        if len( apply_bn_norm ) == 0:
            apply_bn_norm=[ True ] * self.n_lay
        else:
            assert( len( apply_bn_norm ) ) == self.n_lay
            
        # Adjust the tdnn size to include th side info
        for i in range( self.n_lay ):
            self.tdnn_sizes[i][0] += self.side_info_size[i]
            
        if activations == None:        
            self.activations = [tf.nn.relu] * self.n_lay
        else:
            self.activations = activations

        log.info("TDNN architecture: " + str(list(zip(self.tdnn_sizes,self.activations))) )
        
        self.floatX        = floatX
        self.is_test_p     = is_test_p
        self.upd_tdnn      = upd_tdnn
        self.upd_tdnn_spec = upd_tdnn_spec
        self.do_feat_norm  = do_feat_norm
        self.upd_feat_norm = upd_feat_norm


        assert( (len(self.upd_tdnn_spec) == 0) or  (len(self.upd_tdnn_spec) == self.n_lay) ) 
        if (len(self.upd_tdnn_spec) == 0):
            self.upd_tdnn_spec = [True] * self.n_lay
            if (not self.upd_tdnn):
                log.warning("Providing upd_tdnn_spec is meaningless if upd_tdnn=False")
                self.upd_tdnn_spec = [False] * self.n_lay # Just to be extra sure

        
        log.info('Initializing model randomly')
        np.random.seed(17)
                
        bn_decay = 0.95 # Decay rate for batch-norm.

        self.layers = []
        self.layers_upd = []       
        ##############################################################################
        ### Estimate normalization of feats if desired.        
        if (do_feat_norm):
            
            if (not isinstance(it_tr_que, list )):
                log.info("Estimating feature normalization")
                #[X, Y, U, S], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                [X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
                
                ### Apply the normalization    
                mean_feat = np.mean(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
                var_feat  = np.var(feats,axis=(0,1))[np.newaxis,np.newaxis,:]
            elif( len(it_tr_que) == 2 ):
                log.info("Feature normalization initialized with provided mean and std" )                
                mean_feat = it_tr_que[0]
                var_feat  = it_tr_que[1]
            else:
                log.info("Feature normalization initialized with mean=0, std=1" )                
                mean_feat = np.zeros([1,1,self.feat_dim])
                var_feat  = np.ones([1,1,self.feat_dim])
                
            if ( self.upd_feat_norm ):
                log.info("Feature norm will be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =is_test_p, decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
                self.layers_upd +=  [ True ] 
            else:
                log.info("Feature norm will not be updated")
                bn_feats  = tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=mean_feat, var=var_feat,
                                                                         offset=np.zeros_like( mean_feat ),
                                                                         scale=np.ones_like( var_feat ),
                                                                         is_test =tf.constant(True), decay=bn_decay,
                                                                         variance_epsilon=0.0, floatX=self.floatX )
                self.layers_upd +=  [ False ] 
            self.layers +=  [bn_feats] 
    
        ##########################################################################        
        ### Layers 

        for i in range( self.n_lay ):
            assert ( len(tdnn_sizes[i])==4 )

            if self.upd_tdnn and self.upd_tdnn_spec[i]:
                is_test_b_pool = self.is_test_p
            else:
                is_test_b_pool = tf.constant(True)
            
                
            ## Append TDNN layer
            para = tensorflow_code.initializers.init_params_simple_he_uniform_full_spec( [ tdnn_sizes[i][0:2] ],
                                                                                                 floatX=self.floatX, use_bug=self.use_bug)

            n_step    = tdnn_sizes[i][2]
            step_size = tdnn_sizes[i][3]

            self.layers.append(tensorflow_code.nn_def.tf_tdnn(self.session, weight=para['W_1'], bias=para['b_1'],
                                                                             n_step=n_step,
                                                                             in_dim=( para['W_1'].shape[0] - self.side_info_size[i] ) // n_step, 
                                                                             out_dim=para['W_1'].shape[1],
                                                                             step_size=step_size,
                                                                             floatX=self.floatX, side_info_size=self.side_info_size[i]) )

            # Append the non-linearity
            self.layers.append(activations[i])
            
            # Append Batch-norm layer
            if (apply_bn_norm[i]):
                self.layers.append( tensorflow_code.nn_def.tf_batch_norm_mov_avg(self.session, mean=np.zeros_like(para['b_1']),
                                                                                 var=np.ones_like(para['b_1']),
                                                                                 offset=np.zeros_like(para['b_1']),
                                                                                 scale=np.ones_like(para['b_1']),
                                                                                 is_test =is_test_b_pool, decay=bn_decay,
                                                                                 floatX=self.floatX ) )
                if self.upd_tdnn and self.upd_tdnn_spec[i]:
                    self.layers_upd +=  [ True, True, True ]
                else:
                    self.layers_upd +=  [ False, False, False ]
            else:
                if self.upd_tdnn and self.upd_tdnn_spec[i]:
                    self.layers_upd +=  [ True, True ]
                else:
                    self.layers_upd +=  [ False, False ]
                
        
                                              
    def __call__(self,X1_, S1_=None):
        Y_ = X1_
        for i in range( len(self.layers) ):
            if isinstance( self.layers[i], tensorflow_code.nn_def.tf_tdnn ):
                Y_ = self.layers[i]( Y_, S1_ )
            else:
                Y_ = self.layers[i]( Y_ )
                                                          
        return Y_ 

               
    def get_parameters(self):
        params_ = []
        #if self.upd_tdnn:
        for l in self.layers:                        #+ self.layers_after_pooling:
            if (hasattr(l, 'get_parameters')):       # Need to check this since e.g.relu doesn't have params
                params_ += l.get_parameters()  
                    
        return params_

               
    def get_upd_parameters(self):
        params_ = []
        if self.upd_tdnn:
            for i,l in enumerate(self.layers):
                if (hasattr(l, 'get_upd_parameters')) and self.layers_upd[i]:
                    params_ += l.get_upd_parameters() 

        return params_

               
    def get_l2_reg(self, reg=0.0):
        l2_reg = 0.0
        if (reg > 0.0):
            for l in self.layers:
                if isinstance(l, tensorflow_code.nn_def.tf_tdnn):
                    l2_reg += l.get_l2_reg( reg )

        return l2_reg
    
