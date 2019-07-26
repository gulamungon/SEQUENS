# Neural network classes for tensorflow. Several were
# translated from the Theano equivalents.

import numpy as np
import tensorflow as tf
from utils.misc import get_logger
log = get_logger()

import inspect



class tf_batch_norm(object):

  def __init__(self,mean, var, offset=0.0, scale=1.0, is_test=tf.constant(True)):

    self.mean_glob_ = tf.Variable(mean, name='mean_glob_bn') 
    self.var_glob_  = tf.Variable(var, name='var_glob_bn') 
    self.offset_    = tf.Variable(offset, name='offset_bn')  
    self.scale_     = tf.Variable(scale, name='scale_bn')   
    self.is_test    = is_test # We want this variable to be provided. Not created here. Because all
                              # batch norm layers (and maybe also dropout etc.) should use the same.
  
  def __call__(self, X_):

    def normalize_train_data():
      mean_batch_, var_batch_ = tf.nn.moments(X_, axes=0, keep_dims=True)
      Y_train_ = tf.nn.batch_normalization(X_, mean_batch_, var_batch_, self.offset_, self.scale_, variance_epsilon=0.001)
      return Y_train_
    
    def normalize_test_data():
      Y_test  = tf.nn.batch_normalization(X_, self.mean_glob_, self.var_glob_, self.offset_, self.scale_, variance_epsilon=0.001)
      return Y_test
      
    return tf.cond(self.is_test, normalize_test_data, normalize_train_data)   
  

  def get_parameters(self):
    return [self.mean_glob_, self.var_glob_, self.offset_, self.scale_]


  def get_parameter_values(self):
    params = []
    with self.session.as_default(): 
      for p in self.get_parameters():
        params += [ p.eval() ]
    return params

  
  def set_parameter_values(self, params):
    for i, p in enumerate( self.get_parameters() ):
      self.session.run( tf.assign( p, params[i] ) )



  
class tf_batch_norm_mov_avg(object):

  def __init__(self, tf_session, mean, var, offset=np.array([0.0], dtype='float64'), scale=np.array([1.0],dtype='float64'),
               decay=0.99, variance_epsilon=0.001, is_test=tf.constant(True), floatX='float32'):
    
    self.floatX= floatX

    self.session    = tf_session
    self.mean_      = tf.Variable(mean.reshape([1,1,-1]).astype(self.floatX), name='mean_bn') 
    self.var_       = tf.Variable(var.reshape([1,1,-1]).astype(self.floatX), name='var_bn') 

    self.offset_    = tf.Variable(offset.astype(self.floatX), name='offset_bn')  
    self.scale_     = tf.Variable(scale.astype(self.floatX), name='scale_bn')
    self.decay      = decay
    self.is_test    = is_test # We want this variable to be provided. Not created here. Because all
                              # batch norm layers (and maybe also dropout etc.) should use the same.
    self.variance_epsilon = variance_epsilon
                              
    if (self.offset_.shape[0].value <= 1 ):
      print "WARNING: A scalar is used for offset_ in batchnorm (not a vector)"
    if (self.scale_.shape[0].value <= 1 ):
      print "WARNING: A scalar is used for scale_ in batchnorm (not a vector)"
      
    
  def __call__(self, X_):

    def normalize_train_data():
      mean_, var_ = tf.nn.moments(X_, axes=[0,1], keep_dims=True)
      #new_mean_ = self.decay * self.mean_ +  ( 1 - self.decay ) * tf.squeeze(mean_)
      #new_var_  = self.decay * self.var_ + ( 1 - self.decay ) * tf.squeeze(var_)
      new_mean_ = self.decay * self.mean_ +  ( 1 - self.decay ) * tf.squeeze(mean_)
      new_var_  = self.decay * self.var_ + ( 1 - self.decay ) * tf.squeeze(var_)
      
      return mean_, var_, new_mean_, new_var_
    
    def normalize_test_data():
      return self.mean_, self.var_, self.mean_, self.var_

    
    #[mean_, var_] =  tf.cond(self.is_test, normalize_test_data, normalize_train_data)
    [mean_, var_, new_mean_, new_var_] =  tf.cond(self.is_test, normalize_test_data, normalize_train_data)
    mean_ass_op = tf.assign(self.mean_, new_mean_  )
    var_ass_op  = tf.assign(self.var_, new_var_ )
    
    with tf.control_dependencies( [mean_ass_op, var_ass_op  ] ):
      normed_data = tf.nn.batch_normalization(X_, mean_, var_, self.offset_, self.scale_, self.variance_epsilon)
      #normed_data = tf.nn.batch_normalization(X_, mean_, tf.maximum(var_, self.variance_epsilon**2 * tf.ones_like(var_)), self.offset_, self.scale_, 0.0)
      
    return normed_data  #, mean_, var_,  
  

  def get_parameters(self):
    return [self.mean_, self.var_, self.offset_, self.scale_]

  def get_upd_parameters(self):
    return [self.offset_, self.scale_]
  
  def get_parameter_values(self):
    params = []
    with self.session.as_default(): 
      for p in self.get_parameters():
        params += [ p.eval() ]
    return params
 
  def set_parameter_values(self, params):
    for i, p in enumerate( self.get_parameters() ):
      self.session.run( tf.assign( p, params[i] ) )

  
  
class tf_dct_feat_proc( object ):

  def __init__(self, feat_dim=60, left_ctx=15, right_ctx=15, dct_basis=6, floatX='float32'):

    import pytel.features
    #import numpy as np
    self.floatX    = floatX
    self.left_ctx  = left_ctx
    self.right_ctx = right_ctx
    self.dct_basis = dct_basis
    dct_xform      =  pytel.features.dct_basis(dct_basis, left_ctx+right_ctx+1)
    dct_xform[0]   = np.sqrt(2./(left_ctx+right_ctx+1)) # Just to be the same as Franta

    # No flipping compared to the Theano version. This must be float32.
    hamming_dct = (dct_xform*np.hamming(left_ctx+right_ctx+1)).T.astype('float32')

    self.H_  = tf.constant(hamming_dct) #T.constant(hamming_dct.T[:,np.newaxis,:,np.newaxis])

  def __call__(self, X_p ):
    X1_         = X_p[:,0:-1]
    V1_         = X_p[:, -1]
    X1_         = tf.expand_dims(X1_,0)
    X1_         = tf.expand_dims(X1_,-1)

    # If we use float64, we need to cast to float32 for this operation
    if self.floatX == 'float64':
      X1_         = tf.cast(X1_, 'float32')
    Out_        = tf.nn.convolution(X1_, self.H_[:,np.newaxis,np.newaxis,:],padding='VALID',data_format='NHWC')
    if self.floatX == 'float64':
      Out_        = tf.cast(Out_, 'float64')
    Out_        = tf.reshape(Out_,(-1, 360) )

    V1_         = tf.where(tf.equal(V1_[15:-15], 1))
    Out_        = tf.gather(Out_, V1_[:,0] )

    return  Out_  





class tf_tdnn( object ):

  def __init__(self, tf_session, weight, bias, in_dim, n_step, out_dim, step_size=1, floatX='float32', side_info_size =0):

    self.session   = tf_session
    self.floatX    = floatX
    self.step_size = step_size
    self.side_info_size = side_info_size
    if (self.side_info_size > 0):
      log.info("Using side_info_size %d in TDNN", self.side_info_size)
      self.weight    = tf.Variable(weight[:-self.side_info_size, : ].astype('float32'), 'weight_tdnn')
      self.side_vec  = tf.Variable(weight[ -self.side_info_size:,: ].astype('float32'), 'side_info_vector_tdnn')
    else:
      self.weight    = tf.Variable(weight.astype('float32'), 'weight_tdnn')
    self.bias      = tf.Variable(bias.astype(floatX), 'bias_tdnn')
    self.n_step    = n_step
    self.out_dim   = out_dim
    self.in_dim    = in_dim
    
    
  def __call__(self, X_p, S_p=None):
      
    X1_ = X_p[:,:,:]
    #X1_ = tf.expand_dims(X1_,0)
    X1_ = tf.expand_dims(X1_,-1)

    if ((self.side_info_size) > 0 and (S_p==None) ):
      log.error("Context variable not provided in call to TDNN layer")
    if ((self.side_info_size) == 0 and (S_p!=None) ):
      log.warning("Context variable provided in call to TDNN layer but not in its initialization. Will not be used")

    
    # If we use float64, we need to cast to float32 for this operation
    if self.floatX == 'float64':
      X1_         = tf.cast(X1_, 'float32')

    Out_        = tf.nn.convolution(X1_, tf.reshape(self.weight, [self.n_step, self.in_dim, 1, self.out_dim]), # [5,23,1,512]
                                    dilation_rate=[self.step_size, 1], padding='VALID', data_format='NHWC')    #[self.step]
    if self.floatX == 'float64':
      Out_        = tf.cast(Out_, 'float64')

    Out_        = tf.squeeze(Out_, axis=[2]) + self.bias

    if (self.side_info_size > 0):
      Out_ += tf.expand_dims(tf.tensordot(tf.cast(S_p, self.floatX), self.side_vec, axes=1 ), 1)   # [B, side_info_size, Out_size  ] . [B, side_info_size  ]    

    Out_ = tf.identity(Out_, name="TDNN_out")  # Give it a name to help debugging  

    return  Out_  


  def get_parameters(self):
    if (self.side_info_size > 0):
      return [self.weight, self.side_vec, self.bias]
    else:
      return [self.weight, self.bias]

  get_upd_parameters = get_parameters

  
  def get_parameter_values(self):
    params = []
    with self.session.as_default(): 
      for p in self.get_parameters():
        params += [ p.eval() ]
    return params

  def set_parameter_values(self, params):
    for i, p in enumerate( self.get_parameters() ):
      self.session.run( tf.assign( p, params[i] ) )

  def get_l2_reg(self, reg_para):
    return reg_para * reduce(lambda x,y: x+y, [ tf.reduce_sum(p**2) for p in self.get_parameters()  ] )

# Self attention like, NEC and JHU
class tf_self_att_simple( object ):

  def __init__(self, tf_session, in_dim, out_dim, weight_1, weight_2, bias_1=None, bias_2=None, activation = tf.nn.relu,
               n_step=1, step_size=1, floatX='float32'):

    self.session   = tf_session
    self.floatX    = floatX
    self.step_size = step_size
    self.weight_1    = tf.Variable(weight_1.astype('float32'), 'weight_1_tdnn_att')
    self.weight_2    = tf.Variable(weight_2.astype('float32'), 'weight_2_tdnn_att')

    if np.all(bias_1) != None:
      self.bias_1    = tf.Variable(bias_1.astype(floatX), 'bias_tdnn_att')
    else:
      self.bias_1    = None

    if np.all(bias_2) != None:
      self.bias_2    = tf.Variable(bias_2.astype(floatX), 'bias_tdnn_att')
    else:
      self.bias_2    = None
      
    self.n_step    = n_step
    self.out_dim   = out_dim
    self.in_dim    = in_dim
    self.activation = activation

    log.info("Attention activation: " + str(self.activation) )
    
  def __call__(self, X_p ):
    X1_ = X_p[:,:,:]               #[N, T, d ], T= Time, c=context
    X1_ = tf.expand_dims(X1_,-1)   #[N, T, d, 1]  == [batchsize] + [input_spatial_shape] + [in_channels]

    # If we use float64, we need to cast to float32 for this operation
    if self.floatX == 'float64':
      X1_ = tf.cast(X1_, 'float32')

    # Filter (weight_1 ) has shape #[n_step, d, 1, 512]  == [spatial_filter_shape] + [in_channels] + [out_channels]  
    Out_ = tf.nn.convolution(X1_, tf.reshape(self.weight_1, [self.n_step, self.in_dim, 1, self.out_dim]), # [5,23,1,512]
                             dilation_rate=[self.step_size, 1], padding='VALID', data_format='NHWC')    #[self.step]
    
    if self.floatX == 'float64':
      Out_ = tf.cast(Out_, 'float64')     # [N, T-c, 1, 512]

    Out_ = tf.squeeze(Out_, axis=[2])     # [N, T-c, 512]
    
    if np.all(self.bias_1) != None:
      Out_ += self.bias_1                   # Make sure bias has shape [1,1,512] ??

    if self.activation != None:
      Out_ = self.activation(Out_)
    
    Out_ = tf.tensordot(Out_, self.weight_2, axes=1)

    if np.all(self.bias_2) != None:
      Out_ += self.bias_2                   # Make sure bias has shape [1,1,512] ??
    
    return  tf.transpose(tf.nn.softmax(tf.transpose(Out_,[0,2,1])),[0,2,1])
    #return  tf.transpose(tf.nn.softmax(tf.transpose(Out_[0,2,1])),[0,2,1])  


  def get_parameters(self):
    para_ = [self.weight_1]
    if np.all(self.bias_1) != None:
      para_.append(self.bias_1)
    para_.append(self.weight_2)
    if np.all(self.bias_2) != None:
      para_.append(self.bias_2)
    
    return para_

  get_upd_parameters = get_parameters

  
  def get_parameter_values(self):
    params = []
    with self.session.as_default(): 
      for p in self.get_parameters():
        params += [ p.eval() ]
    return params

  def set_parameter_values(self, params):
    for i, p in enumerate( self.get_parameters() ):
      self.session.run( tf.assign( p, params[i] ) )

  def get_l2_reg(self, reg_para):
    return reg_para * reduce(lambda x,y: x+y, [ tf.reduce_sum(p**2) for p in self.get_parameters()  ] )

  def get_att_reg():
    pass
      
class tf_ff_nn( object ):

  def __init__(self, tf_session, param_dict, act_fkns =[], name='ff_nn', floatX='float32'):

    self.floatX=floatX
    
    self.session = tf_session
    self.weights_ = []
    self.biases_  = []
    self.n_layers = ( len(param_dict) - ('input_mean' in param_dict) - ('input_std' in param_dict) )/2

    if ( act_fkns == [] ):
      self.act_fkns = [tf.nn.relu] * (self.n_layers - 1 )
    else:            
      self.act_fkns = act_fkns

    # Mean
    if ( 'input_mean' in param_dict ):
      self.mean_ = tf.Variable(param_dict['input_mean'].astype(self.floatX), name='input_mean')
    else:
      self.mean_ = None

    # Standard deviation 
    if ( 'input_std' in param_dict ):
      self.std_ = tf.Variable(param_dict['input_std'].astype(self.floatX), name='input_std')
    else:
      self.std_ = None

    # Weights and biases
    for ii in range( self.n_layers ):
        self.weights_ += [ tf.Variable( param_dict[ 'W_'+str(ii + 1) ].astype(self.floatX), name='W_'+str( ii + 1 )) ]
        self.biases_  += [ tf.Variable( param_dict[ 'b_'+str(ii + 1) ].astype(self.floatX), name='b_'+str( ii + 1 )) ]


  def __call__(self, X_ ):
    Y_ = X_

    if ( self.mean_ ):
      Y_        = Y_ - self.mean_

    if ( self.std_ ):
      Y_        = Y_ / self.std_

    # Go through the affine transforms  
    for ii in range( self.n_layers ):
        Y_ = tf.tensordot(Y_, self.weights_[ii], axes=1 ) + self.biases_[ii]

        # Apply the non-linearity if there is still one available
        if ( len( self.act_fkns ) > ii  ):
          Y_ = self.act_fkns[ii](  Y_ )

    return Y_


  def get_parameters(self):
    params_ = []
    if self.mean_:
      params_ += [ self.mean_ ]
      
    if self.std_:
      params_ += [ self.std_ ]

    params_ += self.weights_
    params_ += self.biases_

    return params_

  get_upd_parameters = get_parameters

  def get_parameter_values(self):
    params = []
    with self.session.as_default(): 
      for p in self.get_parameters():
        params += [ p.eval() ]
    return params

  
  def set_parameter_values(self, params):
    for i, p in enumerate( self.get_parameters() ):
      self.session.run( tf.assign( p, params[i] ) )

      
  def get_l2_reg(self, reg_para):
    return reg_para * reduce(lambda x,y: x+y, [ tf.reduce_sum(p**2) for p in self.get_parameters()  ] )     

  
  def summary(self):

    if self.mean_:
      print "* Mean: " + str(self.mean_.get_shape() )

    if self.mean_:
      print "* Std: " + str( self.std_.get_shape() )

    # Go through the affine transforms  
    for ii in range( self.n_layers ):

      w = self.weights_[ii]
      b = self.biases_[ii]
      print "* Affine transform" 
      print "   Weight: " + str(w.get_shape())

      print "   Bias:   " + str(b.get_shape())

      # Print the non-linearity if available
      if ( len( self.act_fkns ) > ii  ):
        print "* Non-linearity: " + str( self.act_fkns[ii] )

# Looping over utterances to create embeddings. Not yet used to TF so
# some unclear points:
# 1, Is it better to include X_ and C_ as looping variable (as it it in Theano)
# 2, Is it better to create full embedding matrix before looping then asssign
#    in the loop? Nothing like Thenano's set subtensor exist so a bit messy...
#    IF WE WANT TO USE parallel_iterations > 1, WE NEED TO DO THIS TO PRESERVE
#    ORDER.
# 3, Why TF can't infer shape of Z? See "*".
# 4, For setting shape_invariants we can't use TF tensors, i.e., this needs to
#    specified at the time creating the graph??? Not sure if this is the case.
class tf_pool( object ):

  def __init__(self, nn,  pool_fkn =lambda x: tf.reduce_mean(x, axis=0), output_size=[], loop_swap_memory=False, floatX='float32', l_ctx=0, r_ctx=0):

    self.l_ctx = l_ctx
    self.r_ctx = r_ctx
    
    self.floatX=floatX
    
    if not isinstance(nn, list):
      self.nn = [ nn ]
    else:
      self.nn = nn

    self.pool_fkn         = pool_fkn
    self.loop_swap_memory = loop_swap_memory

    # Not that this automatic assignment will not always work, e.g., in
    # the case when the pool function outputs both means and stds. So it's
    # better to set it explicitly as input argument.
    if ( output_size == [] ):
      self.output_size = nn_pool.nn[-1].biases_[-1].shape.as_list()[0]
    else:
      self.output_size = output_size

    
  def __call__(self, X_, C_, S_=None):

    # This is executed in each loop
    def body(M_, i_):
      Z_ = tf.slice(X_, begin =[0, C_[i_] + self.l_ctx, 0], size=[-1, C_[i_+1]-C_[i_] - self.l_ctx - self.r_ctx,-1], name='pool_loop_slice')
      if len(inspect.getargspec(self.pool_fkn)[0]) == 2:
        log.info("Pool function expects 2 input arguments. Assume org features should be used in addition to processed ones.")
        Z_org_ = Z_
      
      # Concatenate the nns
      for j in range(0, len(self.nn) ):
        if ( (S_ != None) and isinstance( self.nn[j], tf_tdnn ) ):
          log.info("Providing context to TDNN (%d) in pooling", j)
          Z_ = self.nn[j]( Z_, S_)
        else:
          Z_ = self.nn[j]( Z_ )
          
      if len(inspect.getargspec(self.pool_fkn)[0]) == 1:
        Z_ = self.pool_fkn( Z_ )
      elif len(inspect.getargspec(self.pool_fkn)[0]) == 2:
        Z_ = self.pool_fkn( Z_, Z_org_ )
      else:
        log.error("Pool function expects wrong number of varaibles.")
        
      Z_.set_shape([1,1,self.output_size])   # * Seems we TF can not infer the shape although this should be possible. Why?
      Z_ = tf.concat([M_, Z_ ], axis=0)
      return [Z_, tf.add(i_, 1) ]

    # Initial tensors for counter, i, and embeddings, M,
    i0_ = tf.constant(0, dtype='int32', name='pool_loop_index')
    Y0_ = tf.zeros([0, 1, self.output_size], dtype=self.floatX )
    
    M_final_ = tf.while_loop(cond =lambda M_,  i_: tf.less(i_, tf.shape(C_)[0] -1),
                             body=body, loop_vars=[Y0_, i0_ ],
                             shape_invariants=[tf.TensorShape([None, 1, self.output_size]), i0_.get_shape()],
                             parallel_iterations=1, swap_memory=self.loop_swap_memory)[0]
    
    return( M_final_ )






  

### Class for generalized pooling  
# The variance_epsilon is needed to avoid nans in some situations
#def mean_std(x, axes=0, variance_epsilon=1e-8):
#  mean_, var_ = tf.nn.moments(x, axes=axes, keep_dims=True)
#  return ( tf.concat([mean_, tf.sqrt(var_ + variance_epsilon),], axis=-1) )    

class gen_pool_layer( object ):

  # Parameters
  # p:             Vector with values of p where no mean is subtracted from the variable
  # p_mean_sub:    Vector with values of p where mean IS subtracted from the variable
  # p_count:       Power to which we raise the counts (n). Dim(p_count) = Dim(p_mean_sub) + Dim(p).
  #                None neans this option is not used.
  # p_extra_count: 
  def __init__(self, tf_session, p1, p2, p1_count=None, p2_count=None,
               p_extra_count_1=None, p_extra_count_2=None, floatX='float32', epsilon=1e-8):
    self.session   = tf_session
    self.floatX    = floatX

    self.epsilon = epsilon
    
    self.init_p1              = p1
    self.init_p2              = p2
    self.init_p1_count        = p1_count
    self.init_p2_count        = p2_count
    self.init_p_extra_count_1 = p_extra_count_1
    self.init_p_extra_count_2 = p_extra_count_2

    if np.all(self.init_p1) != None:
      log.info("P1 powers are used")      
      self.p1_          = tf.Variable(self.init_p1.astype( self.floatX), 'gen_pool_layer_p1')
    else:
      self.p1_          = None
      assert(np.all(self.init_p1_count) == None)
      
    if np.all(self.init_p2) != None:
      log.info("P1 powers are used")      
      self.p2_          = tf.Variable(self.init_p2.astype( self.floatX), 'gen_pool_layer_p2')
    else:
      self.p2_          = None
      assert(np.all(self.init_p2_count) == None)

    if np.all(self.init_p1_count) != None:
      assert( len(p1_count ) == len(p1) )
      log.info("P1 count powers are used")
      self.p1_count_    = tf.Variable(self.init_p1_count.astype( self.floatX ), 'gen_pool_layer_p1_counts')
    else:
      self.p1_count_    = None

    if np.all(self.init_p2_count) != None:
      assert( len(p2_count ) == len(p2) )
      log.info("P2 count powers are used")
      self.p2_count_    = tf.Variable(self.init_p2_count.astype( self.floatX ), 'gen_pool_layer_p2_counts')
    else:
      self.p2_count_    = None

    if np.all(self.init_p_extra_count_1) != None:
      log.info("Extra count 1 are used. Number of them: " + str(len(self.init_p_extra_count_1)))
      self.p_extra_count_1_    = tf.Variable(self.init_p_extra_count_1.astype( self.floatX ), 'gen_pool_layer_p_extra_count_1')
    else:
      self.p_extra_count_1_    = None

    if np.all(self.init_p_extra_count_2) != None:
      log.info("Extra count 2 are used. Number of them: " + str(len(self.init_p_extra_count_2)))
      self.p_extra_count_2_    = tf.Variable(self.init_p_extra_count_1.astype( self.floatX ), 'gen_pool_layer_p_extra_count_2')
    else:
      self.p_extra_count_2_    = None

      
  def __call__(self, X_,axis=0):
                                  
    # The counts
    n_  = tf.cast(tf.shape(X_)[axis], self.floatX)  # Need to be vector if we want to raise to learnable parameter                               
    Sg_ = tf.sign(X_)

    if ( self.p1_count_ == None):
      n1_ = n_
    else:
      n1_ = tf.pow(n_, self.p1_count_) 

    if ( self.p2_count_ == None):
      n2_ = n_
    else:
      n2_ = tf.pow(n_, self.p2_count_) 

    
    # The standard calculations
    mean_, var_ = tf.nn.moments(X_, axes=axis, keep_dims=True)

    if (self.p1_ == None):
      log.info ("Standard mean calculation will be used")
      S1_ = mean_
    else:
      log.info ("General mean calculation will be used")
      Xa_     = tf.abs(X_)
      Xap_    = tf.pow(Xa_ + self.epsilon, self.p1_)
      Xaps_   = Sg_* Xap_
      Xapss_  = tf.reduce_sum(Xaps_, axis=axis, keep_dims=True)/n1_ 
      Sg2_    = tf.sign( Xapss_ )
      Xapssa_ = tf.abs( Xapss_ )
      S1_     = Sg2_ * tf.pow(Xapssa_ + self.epsilon, 1/self.p1_)
      
    if (self.p2_ == None):
      log.info ("Standard standard deviation calculation will be used")
      S2_ = tf.sqrt(var_ + self.epsilon)
    else:
      log.info ("General standard deviation calculation will be used")
      m_    = tf.reduce_mean(X_, axis=axis, keep_dims=True)
      Xma_  = tf.abs(X_-m_)
      Xmap_ = tf.pow(Xma_ + self.epsilon, self.p2_)  
      S2_   = tf.pow(tf.reduce_sum(Xmap_, axis=axis, keep_dims=True)/n2_ + self.epsilon, 1/self.p2_)

    
    if (self.p_extra_count_1_ == None):
      out_ = tf.concat([S1_,S2_], axis=-1)
    else:
      #out_ = tf.expand_dims(tf.expand_dims(tf.pow(n_, self.p_extra_count_1_), axis=0), axis=0)
      #out_ = tf.expand_dims(tf.expand_dims(tf.nn.relu(n_ -  self.p_extra_count_1_), axis=0), axis=0)
      out_ = tf.expand_dims(tf.expand_dims(1-150*self.p_extra_count_1_/( 150*self.p_extra_count_1_ + n_ ), axis=0), axis=0)      
      out_ = tf.tile(out_, [ tf.shape(S1_)[0], tf.shape(S1_)[1], 1] )
      out_ = tf.concat( [S1_,S2_, out_], axis=-1)
    
    return out_ #, [n_, self.p_extra_count_1_, tf.nn.relu(n_ -  self.p_extra_count_1_)]

                                                                    
  def get_parameters(self):
    params_ = []
    if (self.p1_ != None):
      params_.append(self.p1_)
    if (self.p2_ != None):
      params_.append(self.p2_)
    if ( self.p1_count_ != None):
      params_.append(self.p1_count_)
    if ( self.p2_count_ != None):
      params_.append(self.p2_count_)
    if (self.p_extra_count_1_ != None):
      params_.append( self.p_extra_count_1_ )
    if (self.p_extra_count_2_ != None):
      params_.append( self.p_extra_count_2_ )
    return params_


  def get_parameter_values(self):
    params = []
    with self.session.as_default(): 
      for p in self.get_parameters():
        params += [ p.eval() ]
    return params

  
  def set_parameter_values(self, params):
    for i, p in enumerate( self.get_parameters() ):
      self.session.run( tf.assign( p, params[i] ) )

      # Should reg to 1,2?...    
  #def get_l2_reg(self, reg_para):
  #  return reg_para * reduce(lambda x,y: x+y, [ tf.reduce_sum(p**2) for p in self.get_parameters()  ] )     

  

###################################################################################
### Old stuff
"""
class tf_batch_norm_mov_avg_1(object):

  def __init__(self, tf_session, mean, var, offset=np.array(0.0, dtype='float32'), scale=np.array(1.0,dtype='float32'), decay=0.99, is_test=tf.constant(True)):

    self.session    = tf_session
    self.mean_      = tf.Variable(mean, name='mean_bn') 
    self.var_       = tf.Variable(var, name='var_bn') 
    self.offset_    = tf.Variable(offset, name='offset_bn')  
    self.scale_     = tf.Variable(scale, name='scale_bn')
    self.decay      = decay
    self.is_test    = is_test # We want this variable to be provided. Not created here. Because all
                              # batch norm layers (and maybe also dropout etc.) should use the same.

    if (np.array(self.offset_).shape == () ):
      print "WARNING: A scalar is used for offset_ in batchnorm (not a vector)"
    if (np.array(self.scale_).shape == () ):
      print "WARNING: A scalar is used for scale_ in batchnorm (not a vector)"
      

  def __call__(self, X_):

    def normalize_train_data():
      mean_, var_ = tf.nn.moments(X_, axes=0, keep_dims=True)
      mean_ass_op = tf.assign(self.mean_, tf.squeeze(mean_))
      var_ass_op  = tf.assign(self.var_, tf.squeeze(var_))
      with tf.control_dependencies( [mean_ass_op, var_ass_op] ):
        with tf.control_dependencies( [ self.ema.apply( [self.mean_, self.var_] ) ] ):
          Y_train_ = tf.nn.batch_normalization(X_, mean_ass_op, var_ass_op, self.offset_, self.scale_, variance_epsilon=0.001)
      return Y_train_
    
    def normalize_test_data():
      Y_test  = tf.nn.batch_normalization(X_, self.ema.average(self.mean_), self.ema.average(self.var_),
                                          self.offset_, self.scale_, variance_epsilon=0.001)
      return Y_test
      
    return tf.cond(self.is_test, normalize_test_data, normalize_train_data)   
  

  def get_parameters(self):
    return [self.ema.average(self.mean_), self.ema.average(self.var_), self.offset_, self.scale_]

  def get_parameter_values(self):
    params = []
    with self.session.as_default(): 
      for p in self.get_parameters():
        params += [ p.eval() ]
    return params
 
  def set_parameter_values(self, params):
    for i, p in enumerate( self.get_parameters() ):
      self.session.run( tf.assign( p, params[i] ) )

  def get_mov_avg_op(self):
    return self.maintain_averages_op

      

class tf_batch_norm_mov_avg_2(object):

  def __init__(self, tf_session, mean, var, offset=np.array(0.0, dtype='float32'), scale=np.array(1.0,dtype='float32'), decay=0.99, is_test=tf.constant(True)):

    self.session    = tf_session
    self.mean_      = tf.Variable(mean, name='mean_bn') 
    self.var_       = tf.Variable(var, name='var_bn') 

    self.offset_    = tf.Variable(offset, name='offset_bn')  
    self.scale_     = tf.Variable(scale, name='scale_bn')
    self.decay      = decay
    self.is_test    = is_test # We want this variable to be provided. Not created here. Because all
                              # batch norm layers (and maybe also dropout etc.) should use the same.

    if (np.array(self.offset_).shape == () ):
      print "WARNING: A scalar is used for offset_ in batchnorm (not a vector)"
    if (np.array(self.scale_).shape == () ):
      print "WARNING: A scalar is used for scale_ in batchnorm (not a vector)"
      
    
  def __call__(self, X_):

    def normalize_train_data():
      mean_, var_ = tf.nn.moments(X_, axes=0, keep_dims=True)
      #mean_ass_op = tf.assign(self.mean_, self.decay * self.mean_ +  ( 1 - self.decay ) * tf.squeeze(mean_)  )
      #var_ass_op  = tf.assign(self.var_, self.decay * self.var_ + ( 1 - self.decay ) * tf.squeeze(var_) )
      #with tf.control_dependencies( [mean_ass_op, var_ass_op  ] ):
      #  Y_train_ = tf.nn.batch_normalization(X_, mean_, var_, self.offset_, self.scale_, variance_epsilon=0.001)
      #return Y_train_, 
      Y_train_  = tf.nn.batch_normalization(X_, mean_, var_, self.offset_, self.scale_, variance_epsilon=0.001) 
      new_mean_ = self.decay * self.mean_ +  ( 1 - self.decay ) * tf.squeeze(mean_)
      new_var_  = self.decay * self.var_ + ( 1 - self.decay ) * tf.squeeze(var_)
      
      return Y_train_, new_mean, mean_var
    
    def normalize_test_data():
      Y_test  = tf.nn.batch_normalization(X_, self.mean_,self.var_, self.offset_, self.scale_, variance_epsilon=0.001)
      return Y_test
      
    return tf.cond(self.is_test, normalize_test_data, normalize_train_data)

  

  def get_parameters(self):
    return [self.mean_, self.var_, self.offset_, self.scale_]

  def get_parameter_values(self):
    params = []
    with self.session.as_default(): 
      for p in self.get_parameters():
        params += [ p.eval() ]
    return params
 
  def set_parameter_values(self, params):
    for i, p in enumerate( self.get_parameters() ):
      self.session.run( tf.assign( p, params[i] ) )

  def get_mov_avg_op(self):
    return self.maintain_averages_op


class tf_tdnn_1( object ):

  def __init__(self, weight, bias, in_dim, n_step, out_dim, step_size=1, floatX='float32'):

    self.floatX    = floatX
    self.step_size = step_size
    self.weight    = tf.Variable(weight, 'weight_tdnn')
    self.bias      = tf.Variable(bias, 'bias_tdnn')
    self.n_step    = n_step
    self.out_dim   = out_dim
    self.in_dim    = in_dim
    
  def __call__(self, X_p ):
    X1_ = X_p[:,:]
    X1_ = tf.expand_dims(X1_,0)
    X1_ = tf.expand_dims(X1_,-1)

    # If we use float64, we need to cast to float32 for this operation
    if self.floatX == 'float64':
      X1_         = tf.cast(X1_, 'float32')

    #Out_        = tf.nn.convolution(X1_, self.weight.reshape([self.n_step, self.in_dim, 1, self.out_dim]),    
    #                                dilation_rate=[self.step_size, 1], padding='VALID', data_format='NHWC')   
    Out_        = tf.nn.convolution(X1_, tf.reshape(self.weight, [self.n_step, self.in_dim, 1, self.out_dim]), # [5,23,1,512]
                                    dilation_rate=[self.step_size, 1], padding='VALID', data_format='NHWC')    #[self.step]
    if self.floatX == 'float64':
      Out_        = tf.cast(Out_, 'float64')
      
    Out_        = tf.squeeze(Out_) + self.bias

    return  Out_  


  def get_parameters(self):
    return [self.weight, self.bias]

  def get_parameter_values(self):
    params = []
    with self.session.as_default(): 
      for p in self.get_parameters():
        params += [ p.eval() ]
    return params

  def set_parameter_values(self, params):
    for i, p in enumerate( self.get_parameters() ):
      self.session.run( tf.assign( p, params[i] ) )

class tf_pool_1( object ):

  def __init__(self, nn,  pool_fkn =lambda x: tf.reduce_mean(x, axis=0), output_size=[], loop_swap_memory=False, floatX='float32', l_ctx=0, r_ctx=0):

    self.l_ctx = l_ctx
    self.r_ctx = r_ctx
    
    self.floatX='float32'
    
    if not isinstance(nn, list):
      self.nn = [ nn ]
    else:
      self.nn = nn

    self.pool_fkn         = pool_fkn
    self.loop_swap_memory = loop_swap_memory

    # Not that this automatic assignment will not always work, e.g., in
    # the case when the pool function outputs both means and stds. So it's
    # better to set it explicitly as input argument.
    if ( output_size == [] ):
      self.output_size = nn_pool.nn[-1].biases_[-1].shape.as_list()[0]
    else:
      self.output_size = output_size

    
  def __call__(self, X_, C_):

    # This is executed in each loop
    def body(M_, i_):
      Z_ = tf.slice(X_, begin =[ C_[i_] + self.l_ctx, 0], size=[C_[i_+1]-C_[i_] - self.l_ctx - self.r_ctx,-1], name='pool_loop_slice')
      
      # Concatenate the nns
      for j in range(0, len(self.nn) ):
        Z_ = self.nn[j]( Z_ )
      Z_ = self.pool_fkn( Z_ )
 
      Z_.set_shape([1,self.output_size])   # * Seems we TF can not infer the shape although this should be possible. Why?
      Z_ = tf.concat([M_, Z_ ], axis=0)
      return [Z_, tf.add(i_, 1) ]

    # Initial tensors for counter, i, and embeddings, M,
    i0_ = tf.constant(0, dtype='int32', name='pool_loop_index')
    Y0_ = tf.zeros([0, self.output_size], dtype=self.floatX )
    
    M_final_ = tf.while_loop(cond =lambda M_,  i_: tf.less(i_, tf.shape(C_)[0] -1),
                             body=body, loop_vars=[Y0_, i0_ ],
                             shape_invariants=[tf.TensorShape([None, self.output_size]), i0_.get_shape()],
                             parallel_iterations=1, swap_memory=self.loop_swap_memory)[0]
    
    return( M_final_ )
class gen_pool_layer( object ):

  # Parameters
  # p:          Vector with values of p where no mean is subtracted from the variable
  # p_mean_sub: Vector with values of p where mean IS subtracted from the variable
  # p_count:    Power to which we raise the counts (n). Dim(p_count) = Dim(p_mean_sub) + Dim(p).
  #             [] Means this option is not used.    
  def __init__(self, tf_session, p=1, p_mean_sub=2, p_count=1, axis=1, floatX):
    self.session   = tf_session
    self.floatX    = floatX

    self.axis = axis
    
    self.init_p          = p
    self.init_p_mean_sub = p_mean_sub
    self.init_p_count    = p_count

    self.n_p          = len(p)
    self.n_p_mean_sub = len(p)
    
    self.p          = tf.Variable(self.init_p.astype('float32','gen_pool_layer_p')
    self.p_mean_sub = tf.Variable(self.init_p_mean_sub.astype('float32', 'gen_pool_layer_p_mean_sub')
    self.p_count    = tf.Variable(self.init_p_count.astype('float32', 'gen_pool_layer_p_counts')

    
  def __call__(self, X_):
    X1_ = X_[:, :, 0:self.n_p]                 # The elements where mean IS NOT subtracted.
    X2_ = X_[:, :, self.n_p:self.n_mean_sub]   # The elements where mean IS subtracted.

    # The counts
    n1_  = tf.X1_.shape[self.axis]                              
    n2_  = tf.X2_.shape[self.axis]                                        

        
    X11_  = tf.abs(X1_)
    X111_ = tf.pow(X11_, self.p)
                                  tf.reduce_sum(tfX1, axis=self.axis)                              


                                  n2_  = tf.reduce_sum(X2, axis=self.axis)                              


                                  
    tf.abs(X1_)                            
                                  
  def get_parameters(self):
    params_ = []
    if self.mean_:
      params_ += [ self.mean_ ]
      
    if self.std_:
      params_ += [ self.std_ ]

    params_ += self.weights_
    params_ += self.biases_

    return params_


  def get_parameter_values(self):
    params = []
    with self.session.as_default(): 
      for p in self.get_parameters():
        params += [ p.eval() ]
    return params

  
  def set_parameter_values(self, params):
    for i, p in enumerate( self.get_parameters() ):
      self.session.run( tf.assign( p, params[i] ) )

      
  def get_l2_reg(self, reg_para):
    return reg_para * reduce(lambda x,y: x+y, [ tf.reduce_sum(p**2) for p in self.get_parameters()  ] )     


"""
