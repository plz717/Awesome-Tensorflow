
import tensorflow as tf 
import tensorflow.contrib.slim as slim

#input: a [260,299,299,3] tensor
def extract_feature(input, network_fn):
  
  num_gpus=1
  logits_all=[]
  for i in range(num_gpus):
    with tf.device('/gpu:%d'%i):
      with tf.name_scope('%s_%d' % ('eval', i)) as scope, tf.variable_scope('', reuse=bool(i)):
        '''
        if i<num_gpus-1:
          model_input = tf.slice(input,[32*i,0,0,0],[32,299,299,3])
        else:
          model_input = tf.slice(input,[224,0,0,0],[36,299,299,3])
        '''
        model_input=input
        PreLogits, _ = network_fn(model_input)
        logits_ = slim.flatten(PreLogits)
        logits_all.append(logits_)
  ret_fetures = tf.concat(logits_all,0)
  return ret_fetures




      
