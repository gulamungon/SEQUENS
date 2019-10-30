

from utils.misc import get_logger
log = get_logger()

import tensorflow as tf


# This function returns
# 1, a TF placeholder for learning rate 
# 2, a  F optimizer
# 3, a function for getting the state of the optimizer.
# Having this function to was needed because how to obtain this from different
# TF optimizers are not consistent.
def get_optimizer(opt='SGD',floatX='float32', **kwargs):
    
    lr_p = tf.placeholder(floatX, shape=[], name='lr_p') 

    if(opt == 'SGD'):
        optim = tf.train.GradientDescentOptimizer(learning_rate = lr_p, **kwargs)
        def get_optim_states(optim):
            return []
        return lr_p, optim, get_optim_states
    
    elif(opt == 'Adam'):
        optim = tf.train.AdamOptimizer(learning_rate = lr_p, **kwargs)
        def get_optim_states(optim):
            optim_states = []
            for n in optim.get_slot_names():
                optim_states += list(optim._slots[n].values())
            optim_states += optim._get_beta_accumulators()
            return optim_states
        return lr_p, optim, get_optim_states
        
    elif(opt == 'Momentum'):
        optim = tf.train.MomentumOptimizer(learning_rate = lr_p, **kwargs)
        def get_optim_states(optim):
            optim_states = []
            for n in optim.get_slot_names():
                optim_states += list(optim._slots[n].values())
            return optim_states
        return lr_p, optim, get_optim_states

    elif(opt == 'YF'):
        from tuner_utils import yellowfin 
        optim=yellowfin.YFOptimizer(learning_rate=1.0, **kwargs )
        optim.lr_factor = lr_p   # lr_factor is origally a variable so to change it we would need to use tf.asssign.
                                 # I think it should not cause and problem to change it to a placeholder?
        optim._name = "Yellowfin"
        def get_optim_states(optim):
            optim_states = []
            for n in optim.get_slot_names():
                optim_states += list(optim._optimizer._slots[n].values())
            optim_states += list(optim._moving_averager.variables_to_restore().values())
            return optim_states
        return lr_p, optim, get_optim_states
    else:
        log.error("The optimizer %s is not supported. Please choose one of SGD, Adam, Momemtum and Yellowfin" % opt)
