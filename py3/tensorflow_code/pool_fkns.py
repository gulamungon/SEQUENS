# Pooling functions

import tensorflow as tf
import numpy as np


# The variance_epsilon is needed to avoid nans in some situations
def mean_std(x, axes=0, variance_epsilon=1e-8):
    mean_, var_ = tf.nn.moments(x, axes=axes, keep_dims=True)
    return ( tf.concat([mean_, tf.sqrt(var_ + variance_epsilon),], axis=-1) )    


def mean_std_attention(x, att, axes=0, variance_epsilon=1e-8):
    count_    = tf.reduce_sum(att, axis=[axes], keep_dims=True)
    sum_      = tf.reduce_sum(x*att, axis=[axes], keep_dims=True)
    mean_     = sum_ / count_
    sum_m_2_  = tf.reduce_sum( ((x-mean_)**2)*att, axis=[axes], keep_dims=True )   
    var_      = sum_m_2_ / count_
    return (tf.concat([mean_, tf.sqrt(var_  + variance_epsilon),], axis=-1) )


# Attention is used differently in std calculation
def mean_std_attention_2(x, att, axes=0, variance_epsilon=1e-8):
    count_    = tf.reduce_sum(att, axis=[axes], keep_dims=True)
    sum_      = tf.reduce_sum(x*att, axis=[axes], keep_dims=True)
    mean_     = sum_ / count_
    sum_m_2_  = tf.reduce_sum( ((x*att-mean_)**2), axis=[axes], keep_dims=True )
    var_      = sum_m_2_ / count_
    return (tf.concat([mean_, tf.sqrt(var_  + variance_epsilon),], axis=-1) )


def mean_std_attention_head(x, att, axes=0, variance_epsilon=1e-8):
    x         = tf.expand_dims(x,-1)   #[200,283,500,1]
    att       = tf.expand_dims(att,-2) #[200,283,1 3]
    count_    = tf.reduce_sum(att, axis=[axes], keep_dims=True)
    sum_      = tf.reduce_sum(x*att, axis=[axes], keep_dims=True) #[200,1,500 3]
    mean_     = sum_ / count_
    sum_m_2_  = tf.reduce_sum( ((x-mean_)**2)*att, axis=[axes], keep_dims=True) #[200,1,500 3]    
    var_      = sum_m_2_ / count_
    sh_       = tf.shape(x)    
    mean_     = tf.reshape(mean_, (sh_[0], 1, -1))
    var_      = tf.reshape(var_, (sh_[0], 1, -1))
    return (tf.concat([mean_, tf.sqrt(var_  + variance_epsilon),], axis=-1) )



# Attention is used differently in std calculation
def mean_std_attention_head_2(x, att, axes=0, variance_epsilon=1e-8):
    x         = tf.expand_dims(x,-1)   #[200,283,500,1]
    att       = tf.expand_dims(att,-2) #[200,283,1 3]
    count_    = tf.reduce_sum(att, axis=[axes], keep_dims=True)
    sum_      = tf.reduce_sum(x*att, axis=[axes], keep_dims=True) #[200,1,500 3]
    mean_     = sum_ / count_
    sum_m_2_  = tf.reduce_sum((x*att-mean_)**2, axis=[axes], keep_dims=True) #[200,1,500 3] 
    var_      = sum_m_2_ / count_
    sh_       = tf.shape(x)    
    mean_     = tf.reshape(mean_, (sh_[0], 1, -1))
    var_      = tf.reshape(var_, (sh_[0], 1, -1))
    return (tf.concat([mean_, tf.sqrt(var_  + variance_epsilon),], axis=-1) )





def mean_std_attention_framemerge(x, beta, att, axes=0, variance_epsilon=1e-8, loop_swap_memory=False, no_merge=False, floatX='float32'):

    c_max = 10
    
    if no_merge:
        print("Will not merge frames, only reduce their weight.")
        x_ = x
    else:
        
        bs_ = tf.shape(beta)
        bb_ = tf.concat([ tf.zeros([bs_[0], c_max, 1]), beta], axis=1)
        bbb_= tf.ones(bs_)

        w_  = tf.ones(bs_)

        zs_ = tf.shape(x)
        zz_ = tf.concat([ tf.zeros([zs_[0], c_max, zs_[2]]), x], axis=1)
        zzz_= zz_[:,c_max:,:]

        def body(zzz_,w_,bbb_,i_):
            bbb_ *= (1-bb_[:,c_max-i_-1:-i_-1,:])
            zzz_ += zz_[:,c_max-i_-1:-i_-1,:] * bbb_
            w_  += bbb_
            return zzz_,w_,bbb_, i_+1

        # Initialization
        i0_          = tf.constant(0, dtype='int32', name='pool_loop_index')
        
        x_, w_  = tf.while_loop(cond =lambda zz, ww, bb, ii: tf.less(ii, c_max ),
                                       body=body, loop_vars=[zzz_, w_, bbb_, i0_ ],
                                       shape_invariants=[zzz_.get_shape(), w_.get_shape(), bbb_.get_shape(), i0_.get_shape()],
                                       parallel_iterations=1, swap_memory=True)[0:2]

        x_ = x_ / w_        
                        
    if att == None:
        att = beta 
    else:
        att = att * beta 

    out_ = mean_std_attention(x_, att, axes=axes)
    
    return out_




def mean_std_attention_framemerge_8(x, beta, att, axes=0, variance_epsilon=1e-8, loop_swap_memory=False, no_merge=False, floatX='float32'):

    c_max = 10
    
    if no_merge:
        print("Will not merge frames, only reduce their weight.")
        x_ = x
    else:
        
        bs_ = tf.shape(beta)
        bb_ = tf.concat([ tf.zeros([bs_[0], c_max, 1]), beta], axis=1)
        bbb_= tf.ones(bs_)

        w_  = tf.ones(bs_)

        zs_ = tf.shape(x)
        zz_ = tf.concat([ tf.zeros([zs_[0], c_max, zs_[2]]), x], axis=1)
        zzz_= zz_[:,c_max:,:]

        for i in range(c_max):
            print(i)
            bbb_ *= (1-bb_[:,c_max-i-1:-i-1,:])
            zzz_ += zz_[:,c_max-i-1:-i-1,:] * bbb_
            w_  += bbb_

        x_ = zzz_ / w_
                        
    if att == None:
        att = beta 
    else:
        att = att * beta 

    out_ = mean_std_attention(x_, att, axes=axes)
    
    return out_





# Calculates each step individually. Does not uses prev. step.
def mean_std_attention_framemerge_7(x, beta, att, axes=0, variance_epsilon=1e-8, loop_swap_memory=False, no_merge=False, floatX='float32'):

    if no_merge:
        print("Will not merge frames, only reduce their weight.")
        x_ = x
    else:
        # Initialization
        i0_          = tf.constant(1, dtype='int32', name='pool_loop_index')
        x_a_       = tf.TensorArray(dtype=floatX, size=tf.shape(beta)[1], element_shape=tf.TensorShape([None, 23]) )
        w_a_       = tf.TensorArray(dtype=floatX, size=tf.shape(beta)[1], element_shape=tf.TensorShape([None, 1]) )
        x_         = x[:,0,:]
        w_         = tf.ones([tf.shape(beta)[0], 1])
                              
        x_a_ = x_a_.write(0, x[:,0,:])
        w_a_ = w_a_.write(0, tf.ones([tf.shape(beta)[0], 1]))
        
        def body(i_, x_, w_, x_a_, w_a_ ):

            #x_new_ = (x[:,i_,:] +  x_a_.read(i_ -1)  * ( 1 - beta[:,i_-1,:] )) 
            #w_new_ = (1 +          w_a_.read(i_ -1)  * ( 1 - beta[:,i_-1,:] )) 
            #x_a_ = x_a_.write(i_, x_new_)
            #w_a_ = w_a_.write(i_, w_new_)

            x_ = x[:,i_,:] +  x_  * ( 1 - beta[:,i_-1,:] ) 
            w_ = 1 +          w_  * ( 1 - beta[:,i_-1,:] )

            x_a_ = x_a_.write(i_, x_)
            w_a_ = w_a_.write(i_, w_)
            return [tf.add(i_, 1), x_, w_, x_a_, w_a_]
        
        
        _, _, _, x_a_new_, w_a_new_= tf.while_loop(cond = lambda ii, xx, ww, xa, xw: tf.less(ii, tf.shape(beta)[1]),
                                              body=body, loop_vars=[i0_, x_, w_, x_a_, w_a_],
                                              #shape_invariants=[i0_.get_shape()], #tf.shape(beta)[1], tf.shape(beta)[1],
                                              parallel_iterations=1, swap_memory=loop_swap_memory)
        
        x_ = tf.transpose(x_a_new_.stack() / w_a_new_.stack(), perm=[1,0,2] ) # Output of map has instances in first dim which
                                                                              # is time in our case. So need to switch dims.
    if att == None:
        att = beta 
    else:
        att = att * beta 

    out = mean_std_attention(x_, att, axes=axes)
    return out





# Calculates each step individually. Does not uses prev. step.
def mean_std_attention_framemerge_6(x, beta, att, axes=0, variance_epsilon=1e-8, loop_swap_memory=False, no_merge=False, floatX='float32'):

    c_max = 10
    
    if no_merge:
        print("Will not merge frames, only reduce their weight.")
        x_ = x
    else:
        # Initialization
        i0_          = tf.constant(1, dtype='int32', name='pool_loop_index')
        x_a_       = tf.TensorArray(dtype=floatX, size=tf.shape(beta)[1], element_shape=tf.TensorShape([None, 1500]) )
        w_a_       = tf.TensorArray(dtype=floatX, size=tf.shape(beta)[1], element_shape=tf.TensorShape([None, 1]))
        
        
        def body(i_, x_a_, w_a_ ):
            ii_ = tf.cast(i_, 'int32')
            s_ = tf.maximum(0, ii_ - c_max)
            w_ = tf.cumprod(1-beta[:,s_:ii_,:], reverse=True, axis=axes)
            z_ = tf.reduce_sum(x[:,s_:ii_,:] * w_, axis=axes ) + x[:,ii_,:]   
            w_ = tf.reduce_sum( w_, axis=axes ) + 1
            x_a_ = x_a_.write(i_, z_)
            w_a_ = w_a_.write(i_, w_)
            return [tf.add(i_, 1), x_a_, w_a_]
        
        
        i_, x_a_new_, w_a_new_= tf.while_loop(cond = lambda ii, xx, ww: tf.less(ii, tf.shape(beta)[1]),
                                              body=body, loop_vars=[i0_, x_a_, w_a_],
                                              #shape_invariants=[i0_.get_shape()], #tf.shape(beta)[1], tf.shape(beta)[1],
                                              parallel_iterations=10, swap_memory=loop_swap_memory)
        
        x_a_new_ = x_a_new_.write(0, x[:,0,:])
        w_a_new_ = w_a_new_.write(0, tf.ones([tf.shape(beta)[0], 1]))
        x_ = tf.transpose(x_a_new_.stack() / w_a_new_.stack(), perm=[1,0,2] ) # Output of map has instances in first dim which
                                                                              # is time in our case. So need to switch dims.
    if att == None:
        att = beta 
    else:
        att = att * beta 

    out = mean_std_attention(x_, att, axes=axes)
    return out




# Without using tf loop. Instead expands matrices.
def mean_std_attention_framemerge_5(x, beta, att, axes=0, variance_epsilon=1e-8, loop_swap_memory=False, no_merge=False, floatX='float32'):

    c_max = 10
    
    if no_merge:
        print("Will not merge frames, only reduce their weight.")
        x_ = x
    else:
        
        bs_ = tf.shape(beta)
        bb_ = tf.expand_dims(tf.concat([ tf.zeros([bs_[0], c_max-1, 1]), beta],axis=1),-1)
        bbb_= bb_[:,c_max-1:,:,:]

        is_ = tf.shape(x)
        zz_ = tf.expand_dims(tf.concat([ tf.zeros([is_[0], c_max-1, is_[2]]), x], axis=1), -1)
        zzz_= zz_[:,c_max-1:,:,:]
        for i in range(1,c_max):
            print(i)
            zzz_= tf.concat([zz_[:,c_max-i-1:-i,...], zzz_],axis=3)
            bbb_= tf.concat([bb_[:,c_max-i-1:-i,...], bbb_],axis=3)

        #w_ = tf.cumprod(1 - tf.squeeze(bbb_), reverse=True, axis=2)
        w_ = tf.cumprod(1 - bbb_, reverse=True, axis=3)

        z_new_= tf.reduce_sum(zzz_[:,:-1,:,:] * w_[:,:-1,:,:], axis=3) + x[:,1:,:]
        z_new_= tf.concat([x[:,0:1,:], z_new_], axis=1 )

        w_= tf.concat([tf.ones([is_[0],1,1]), tf.reduce_sum(w_, axis=3)[:,:-1,:]+1],axis=1)
        
        x_ = z_new_ / w_
                        
    if att == None:
        att = beta 
    else:
        att = att * beta 

    out_ = mean_std_attention(x_, att, axes=axes)
    
    return out_

# Calculates each step individually. Does not uses prev. step.
def mean_std_attention_framemerge_4(x, beta, att, axes=0, variance_epsilon=1e-8, loop_swap_memory=False, no_merge=False, floatX='float32'):

    c_max = 10
    
    if no_merge:
        print("Will not merge frames, only reduce their weight.")
        x_ = x
    else:
        def body(i_ ):
            ii_ = tf.cast(i_, 'int32')
            s_ = tf.maximum(0, ii_ - c_max)
            w_ = tf.cumprod(1-beta[:,s_:ii_,:], reverse=True, axis=axes)
            z_ = tf.reduce_sum(x[:,s_:ii_,:] * w_, axis=axes ) + x[:,ii_,:]
            w_ = tf.reduce_sum( w_, axis=axes ) + 1
            return [z_, w_]
        

        i_ = tf.range(start=0.0, limit=tf.cast(tf.shape(x)[1],'float32'), dtype='float32')
        #with tf.device('/cpu:0'):
        x_, w_ = tf.map_fn(fn=body, elems=i_, dtype=[floatX, floatX],
                           parallel_iterations=1, swap_memory=False)

        x_ = tf.transpose(x_ / w_, perm=[1,0,2] ) # Output of map has instances in first dim which is time in our case.
                                                  # So need to switch dims.
    if att == None:
        att = beta 
    else:
        att = att * beta 

    out = mean_std_attention(x_, att, axes=axes)
    return out



# No weight?
def mean_std_attention_framemerge_3(x, beta, att, axes=0, variance_epsilon=1e-8, loop_swap_memory=False, no_merge=False):

    t = 0
    
    if no_merge:
        print("Will not merge frames, only reduce their weight.")
        x_new_ = x
    else:
        #def body(x_, w_, bv_, i_):
        def body(x_, i_):

            x_new_ = tf.concat([x_, x[:, i_: i_ +1, :] ], axis=1)
            return [ x_new_, tf.add(i_, 1) ]

        # Initialization
        i0_          = tf.constant(1, dtype='int32', name='pool_loop_index')
        bv0_          = tf.ones([tf.shape(beta)[0], t+1, 1])
        x_new_       = x[:,0:1,:]
        w_new_       = tf.ones([tf.shape(x)[0],1,1])
        
        x_new_ = tf.while_loop(cond =lambda xx, ii: tf.less(ii, tf.shape(beta)[1] ),
                                       body=body, loop_vars=[x_new_, i0_ ],
                                       shape_invariants=[x.get_shape(), i0_.get_shape()],
                                       parallel_iterations=1, swap_memory=loop_swap_memory)[0]
        x_new_ = x_new_ #/ w_new_
        
    if att == None:
        att = beta 
    else:
        att = att * beta 
    out = mean_std_attention(x_new_, att, axes=axes)
    return out

# Keeps track of t last beta. Multiply by latest beta in each step.
def mean_std_attention_framemerge_2(x, beta, att, axes=0, variance_epsilon=1e-8, loop_swap_memory=False, no_merge=False):

    t   = 10
    
    if no_merge:
        print("Will not merge frames, only reduce their weight.")
        x_new_ = x
    else:
        def body(x_, w_, bv_, i_):

            tt_ = tf.minimum(t, i_)
            
            bv_new_ = tf.concat( [bv_[:,1:,:] *(1- beta[:,i_-1:i_,:]), tf.ones([tf.shape(bv_)[0],1,1])], axis=axes  )
            bv_new_.set_shape([None, t+1, 1])
            x_new_  = tf.reduce_sum(x[:, i_  -tt_ : i_ +1, :] * bv_new_[:,t-tt_:,:], axis=axes, keep_dims=True)            
            x_new_ = tf.concat([x_, x_new_], axis=1)

            w_new_  = tf.reduce_sum(bv_new_[:,t-tt_:,:], axis=axes, keep_dims=True)
            w_new_.set_shape([None,None,1])
            w_new_ = tf.concat([w_, w_new_], axis=1)

            
            return [x_new_, w_new_, bv_new_, tf.add(i_, 1) ]

        # Initialization
        i0_          = tf.constant(1, dtype='int32', name='pool_loop_index')
        bv0_          = tf.ones([tf.shape(beta)[0], t+1, 1])
        x_new_       = x[:,0:1,:]
        w_new_       = tf.ones([tf.shape(x)[0],1,1])

        
        x_new_, w_new_ = tf.while_loop(cond =lambda xx, ww, bb, ii: tf.less(ii, tf.shape(beta)[1] ),
                                       body=body, loop_vars=[x_new_, w_new_, bv0_, i0_ ],
                                       shape_invariants=[x.get_shape(), tf.TensorShape([None,None,1]), bv0_.get_shape(), i0_.get_shape()],
                                       parallel_iterations=1, swap_memory=loop_swap_memory)[0:2]
        x_new_ = x_new_ / w_new_
        
    if att == None:
        att = beta 
    else:
        att = att * beta 
    out = mean_std_attention(x_new_, att, axes=axes)
    return out

# Summming x inside loop, frame by frame the natural way.
def mean_std_attention_framemerge_1(x, beta, att, axes=0, variance_epsilon=1e-8, loop_swap_memory=True, no_merge=False):

    if no_merge:
        print("Will not merge frames, only reduce their weight.")
        x_new_ = x
    else:
        def body(x_, w_, i_):

            #x_new_ = (x[:,i_:i_+1,:] +  x[:,i_ -1:i_,:]  * ( 1 - tf.expand_dims(beta[:,i_-1:i_],2) )) / (2 - tf.expand_dims(beta[:,i_-1:i_],2))
            x_new_ = (x[:,i_:i_+1,:] +  x_[:,i_ -1:i_,:]  * ( 1 - beta[:,i_-1:i_,:] )) #/ (2 - beta[:,i_-1:i_,:])
            x_new_ = tf.concat([x_, x_new_], axis=1)
            w_new_ = (1 +  w_[:,i_ -1:i_,:]  * ( 1 - beta[:,i_-1:i_,:] )) #/ (2 - beta[:,i_-1:i_,:])
            w_new_ = tf.concat([w_, w_new_], axis=1)

            return [x_new_, w_new_, tf.add(i_, 1) ]

        # Initialization
        i0_    = tf.constant(1, dtype='int32', name='pool_loop_index')
        x_new_ = x[:,0:1,:]
        w_new_ = tf.ones([tf.shape(x)[0],1,1])

        x_new_, w_new_ = tf.while_loop(cond =lambda xx, ww, ii: tf.less(ii, tf.shape(beta)[1] ),
                                       body=body, loop_vars=[x_new_, w_new_, i0_ ],
                                       shape_invariants=[x.get_shape(), tf.TensorShape([None,None,1]), i0_.get_shape()],
                                       parallel_iterations=1, swap_memory=loop_swap_memory)[0:2]
        x_new_ = x_new_ / w_new_ 
    if att == None:
        att = beta #tf.expand_dims( beta, 2)                                                                   
    else:
        att = att * beta #tf.expand_dims( beta, 2)                                                                   
        #pass
    out = mean_std_attention(x_new_, att, axes=axes)
    return out 

    

# I experimented with applying instead of adding variance_epsilong but 
# this did not solve the nan problem
"""
def mean_std(x, axes=0):
    mean_, var_ = tf.nn.moments(x, axes=axes, keep_dims=True)
    return ( tf.concat([mean_, tf.sqrt(tf.nn.relu(var_)),], axis=-1) )    
"""


def m_std(x, floatX):
    n = tf.cast(tf.shape(x)[0], floatX)
    s1 = tf.reduce_sum(x, axis=0, keep_dims=True)
    # If n > 0 this gives us the mean/std. If n==0, it gives us 0
    n2 = tf.cond(tf.equal(n, 0), lambda: np.ones(1).squeeze().astype(floatX), lambda: n)
    m = n * s1 / (n2 **2)
    
    s2 = tf.reduce_sum((x - m) **2, axis=0, keep_dims=True)
    s  = tf.sqrt(n * s2  / (n2**2)) 

    return tf.concat([m, s], axis=1)



