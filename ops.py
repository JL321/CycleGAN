import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer

def resnet_layer(h_out, filter_specs):
    
    z = h_out
    for filter_spec in filter_specs:
        z = tf.contrib.layers.instance_norm(tf.contrib.layers.conv2d(z, filter_spec[0], filter_spec[1]))
    
    out = tf.nn.relu(z+h_out)
    
    return out

def resnet_transpose(h_out, scope = None):
    
    if (scope == None):
        z = tf.contrib.layers.conv2d_transpose(h_out, 128, 3)
    else:
        z = tf.contrib.layers.conv2d_transpose(h_out, 128, 3, scope = scope)
    return tf.nn.relu(z+h_out)

def leak_relu(x, leak = 0.2):
    #Leaky relu
    return tf.maximum(x, leak*x) 

def batch_layer(x, units, batch_norm = False):
    z = tf.contrib.layers.fully_connected(x, units, activation_fn = tf.nn.leaky_relu)
    if batch_norm: 
        z = tf.layers.batch_normalization(z)
    return z

def instance_conv(x, units, kernel, stride = 1, activation_fn = tf.nn.relu):
    
    z = tf.contrib.layers.conv2d(x, units, kernel, stride = stride, activation_fn = activation_fn)
    
    return z

def custom_conv2d(x, units, kernel, stride, padding = 'SAME', activation_fn = tf.nn.relu, tag = "_0", batch_norm = False):
    
    #Tag acts as a tag for kernel and bias
    kernel = tf.get_variable("kernel_{}".format(tag), shape = [kernel, kernel, x.shape[-1], units], initializer = xavier_initializer(),
                             trainable = True)
    bias = tf.get_variable("bias_{}".format(tag), shape = [units] , initializer = tf.constant_initializer(0.0), trainable = True) #Initialize as all 0s
    network_out = tf.nn.conv2d(x, spectral_norm(kernel, "kernel_{}".format(tag)), [1, stride, stride, 1], padding)
    network_out = tf.nn.bias_add(network_out, bias)
    
    out = activation_fn(network_out)
    if batch_norm:
        out = tf.contrib.layers.batch_norm(out)
    return out
    
def custom_dense(x, units, batch_norm, activation_fn = tf.nn.relu, tag = "_0"):
    
    weight = tf.get_variable("weight_{}".format(tag), shape = [x.shape[-1], units], initializer = xavier_initializer())
    bias = tf.get_variable("bias_{}".format(tag), shape = [units], initializer = tf.constant_initializer(0.0))
    out = tf.matmul(x, spectral_norm(weight, "weight_{}".format(tag)))+bias
    if (batch_norm):
        out = tf.layers.batch_normalization(out)
    return out

def spectral_norm(weight, weight_n): #Name specification to have a separate u for every individual time step
    
    og_shape = weight.shape
    weight = tf.reshape(weight, (-1, weight.shape[-1])) #Used for the convolutional operator    
    u_shape = [1, weight.shape[-1]]
        
    u = tf.get_variable('u_{}'.format(weight_n), shape = u_shape, initializer = tf.initializers.truncated_normal(), trainable = False)
    u_main = u
    
    v = tf.matmul(u_main, tf.transpose(weight))/tf.norm(tf.matmul(u, tf.transpose(weight)))
    u_main = tf.matmul(v, weight)/tf.norm(tf.matmul(v, weight))
    
    sigma_term = tf.matmul(tf.matmul(v, weight), tf.transpose(u_main))
    normalized = weight / sigma_term 
    
    with tf.control_dependencies([u.assign(u_main)]):
        weight = tf.reshape(normalized, og_shape)  

    
    return weight