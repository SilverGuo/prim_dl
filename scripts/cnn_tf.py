import tensorflow as tf

__all__ = ['weight_variable',
           'bias_variable', 
           'conv2d', 
           'max_pool_2x2']

#==============================================================================
# weight_variable
#==============================================================================
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#==============================================================================
# bias_variable
#==============================================================================
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#==============================================================================
# conv2d
#==============================================================================
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#==============================================================================
# max_pool_2x2
#==============================================================================
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

