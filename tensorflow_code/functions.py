import tensorflow as tf

def tf_reverse_gradient(x):
    return -x + tf.stop_gradient(2*x)
