import tensorflow as tf

def resnet_layer(h_out, filter_specs):
    
    z = h_out
    for filter_spec in filter_specs:
        z = tf.contrib.layers.instance_norm(tf.contrib.layers.conv2d(z, filter_spec[0], filter_spec[1]))
    
    out = tf.nn.relu(z+h_out)
    
    return out

def leak_relu(x, leak = 0.2):
    #Leaky relu
    return tf.maximum(x, leak*x) 

def batch_layer(x, units, batch_norm = True):
    z = tf.contrib.layers.fully_connected(x, units, activation_fn = tf.nn.leaky_relu)
    if batch_norm: 
        z = tf.layers.batch_normalization(z)
    return z

def instance_conv(x, units, kernel, stride = 1, activation_fn = tf.nn.relu):
    
    z = tf.contrib.layers.instance_norm(tf.contrib.layers.conv2d(x, units, kernel, stride = stride, activation_fn = activation_fn))
    
    return z