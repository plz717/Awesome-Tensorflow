# from policy_estimators import *
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import os
import collections
import time
import numpy as np
np.set_printoptions(threshold=np.nan)

import tensorflow as tf
import math

from tensorflow.python import pywrap_tensorflow
from losses import CrossEntropyLoss, CrossEntropyBetweenEachSample


slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('train_data_dir', '/mnt/lustre/DATAshare/webvision/images_lables_features/tfrecord',
                       'path to dataset in TFRecord format (aka Example protobufs). If not specified, synthetic data will be used.')
tf.flags.DEFINE_string('validation_data_dir', '/mnt/lustre/DATAshare/webvision_tfrecord',
                       'path to dataset in TFRecord format (aka Example protobufs). If not specified, synthetic data will be used.')
tf.flags.DEFINE_string("data_set", "WebVision",
                       "Which dataset to use for the model. datasets are defined "
                       "in datasets.py.")
tf.flags.DEFINE_integer('input_data_type', tf.float32, 'input data type for training')
tf.flags.DEFINE_string('preprocessing_train', 'webvision_preprocessing_by_plz', 'which preprocessing to use')
tf.flags.DEFINE_string('preprocessing_validation', 'webvision_reimplement_preprocessing', 'which preprocessing to use')

tf.flags.DEFINE_integer('frame_size', 224, 'frame_size.')
tf.flags.DEFINE_integer('channels', 3, 'channels of input data')
tf.flags.DEFINE_integer('batch_size', 64, 'total batch size on all compute device')
tf.flags.DEFINE_string('data_format', 'NHWC', 'order of input data, may be NCHW or NHWC')

tf.flags.DEFINE_string('feature_extractor_name', 'inception_v4', 'name of feature extractor')
tf.flags.DEFINE_string('ckpt_file_dir', '/mnt/lustre/DATAshare/model-zoo/', 'path to the pretrained model used for feature extraction')

tf.flags.DEFINE_string("base_classifier_name", "inception_resnet_v2", "Basic classifier model selection. ")
tf.flags.DEFINE_integer('max_episodes', 24000, 'number of episodes to run for training policy network')
tf.flags.DEFINE_integer('episode_time_steps', 20, 'number of time steps in each episode')
tf.flags.DEFINE_integer('validation_iterations', 4, 'iterations in validation process')
tf.flags.DEFINE_integer('validation_BATCH_SIZE', 4, 'batch_size in validation process')
tf.flags.DEFINE_string('classifier_ckpt_files', '/mnt/lustre/panglinzhuo/rl_noisy_data/classifier_ckpt_files/lr_0.005/', 'classifier_ckpt_files')
tf.flags.DEFINE_string('actor_ckpt_files', '/mnt/lustre/panglinzhuo/rl_noisy_data/actor_ckpt_files/', 'actor_ckpt_files')

tf.flags.DEFINE_boolean('actor_work', False, 'whether or not the policy network is working')
tf.flags.DEFINE_integer('Actor_gpu_num', 3, """Initial learning rate for training.""")
tf.flags.DEFINE_integer('actions_dim', 2, 'output channels of policy network')
tf.flags.DEFINE_integer('layer_nodes', [1000, 500, 100, 2], 'nodes of middle layers in policy network')
tf.flags.DEFINE_float('high', 0.85, 'highest accuracy threshold during training policy network')
tf.flags.DEFINE_float('low', 0.5, 'lowest accuracy threshold during training policy network')
tf.flags.DEFINE_integer('T', 1000, 'After T steps, the accuracy threshold reachs its maximum')
tf.flags.DEFINE_float('dis_factor', 0.9, 'discount factor during future reward calculation')
tf.flags.DEFINE_string('log_dir', '/mnt/lustre/panglinzhuo/rl_noisy_data/logs/', 'dir to store operations summary')
tf.flags.DEFINE_float('weight_decay', 0.00001, """Weight decay factor for training.""")
tf.flags.DEFINE_float('learning_rate', 0.028284271, """Initial learning rate for training.""")
tf.flags.DEFINE_float('learning_rate_decay_factor', 0.94, """Initial learning rate for training.""")
tf.flags.DEFINE_float('momentum', 0.9, """Momentum for training classifier.""")
tf.flags.DEFINE_string('optimizer', 'rmsprop', """optimizer for training classifier.""")
tf.flags.DEFINE_float('rmsprop_decay', 0.9, """Decay term for RMSProp.""")
tf.flags.DEFINE_float('rmsprop_momentum', 0.9, """Momentum in RMSProp.""")
tf.flags.DEFINE_float('rmsprop_epsilon', 1.0, """Epsilon term for RMSProp.""")
tf.flags.DEFINE_float('num_epochs_decay', 2, """Initial learning rate for training.""")
tf.flags.DEFINE_integer('Classifier_gpu_num', 1, """Initial learning rate for training.""")
tf.flags.DEFINE_boolean('log_device_placement', False, 'whether or not the policy network is working')


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name)


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name)


def shape(tensor):
    '''
        return the shape of a tensor in tuple format
    '''
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


def BN(input_tensor, axis):
    depth = shape(input_tensor)[-1]
    beta = tf.Variable(tf.constant(
        0.0, shape=[depth]), name='beta', trainable=True)
    gama = tf.Variable(tf.constant(
        1.0, shape=[depth]), name='gama', trainable=True)
    batch_mean, batch_variance = tf.nn.moments(input_tensor, axis)
    normalized_tensor = tf.nn.batch_normalization(
        input_tensor, batch_mean, batch_variance, beta, gama, variance_epsilon=1e-3)
    return normalized_tensor


def fc_mnsit_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu, biases_initializer=tf.constant_initializer(0.1), weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


def cnn(inputs, num_classes, is_training, dropout_keep_prob, scope='cnn'):
    with tf.variable_scope(scope, 'cnn', [inputs]) as sc:
        print("sc.name is:".format(sc.name))
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.fully_connected(net, 100, scope='fc3')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout4')
                net = slim.fully_connected(net, 50, scope='fc5')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
            #  net = slim.conv2d(net, num_classes, [1, 1],
            #                    activation_fn=None,
            #                    normalizer_fn=None,
            #                    biases_initializer=tf.zeros_initializer(),
            #                    scope='fc8')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(
                end_points_collection)
            # if spatial_squeeze:
            #  net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
            #  end_points[sc.name + '/fc8'] = net
            end_points['PreLogits'] = net
            return net, end_points


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, var in grad_and_vars:
            # if var.op.name.startswith(scope):
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        # v = grad_and_vars[16][1]
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class Classifier():
    """The classifier need to be optimized by policy network"""

    def __init__(self, global_step, learning_rate=0.01, num_classes=10, is_training=True):
        self.x = tf.placeholder("float", [None, 28, 28, 1])
        self.y_ = tf.placeholder("float", [None, num_classes])
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.opt = tf.train.RMSPropOptimizer(self.lr, FLAGS.rmsprop_decay, momentum=FLAGS.rmsprop_momentum, epsilon=FLAGS.rmsprop_epsilon)
        self.global_step = global_step

        num_split = FLAGS.Classifier_gpu_num
        x_splits = tf.split(self.x, num_split, 0)
        label_y_splits = tf.split(self.y_, num_split, 0)

        tower_grads = []
        tower_predictions = []
        each_sample_loss_list = []
        with tf.variable_scope("classifier") as scope:
            for i in xrange(num_split):
                with tf.device('/gpu:%d' % i):
                    with slim.arg_scope(fc_mnsit_arg_scope()):
                        PreLogits, endpoints = cnn(
                            x_splits[i], num_classes=10, is_training=True, dropout_keep_prob=0.5)
                    with tf.variable_scope('Logits'):
                        logits_ = slim.flatten(PreLogits)
                        logits_ = slim.fully_connected(
                            logits_, num_classes, activation_fn=None, scope='logits')
                    predictions = tf.nn.softmax(logits_, name='predictions')
                    tower_predictions.append(predictions)

                    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(label_y_splits[i], 1))
                    with tf.name_scope('accuracy'):
                        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    batch_mean_loss_calculator = CrossEntropyLoss()
                    with tf.name_scope('batch_mean_loss'):
                        self.batch_mean_loss = batch_mean_loss_calculator.calculate_loss(predictions, label_y_splits[i])
                    self._loss_summary = tf.summary.scalar('batch_mean_loss', self.batch_mean_loss)

                    # loss between each prediction and each label
                    each_sample_loss_calculator = CrossEntropyBetweenEachSample()
                    each_sample_loss = each_sample_loss_calculator.calculate_loss(predictions, label_y_splits[i])  # 3*1
                    each_sample_loss_list.append(each_sample_loss)

                    # tf.get_variable_scope().reuse_variables()
                    scope.reuse_variables()

                    grads = self.opt.compute_gradients(self.batch_mean_loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        apply_gradient_op = self.opt.apply_gradients(grads, global_step=self.global_step)

        variable_averages = tf.train.ExponentialMovingAverage(
            0.99, self.global_step)
        trainable_variables = [v for v in tf.trainable_variables() if v.op.name.startswith('classifier')]

        variables_averages_op = variable_averages.apply(trainable_variables)

        # Group all updates to into a single train op.
        self.train_op = tf.group(apply_gradient_op, variables_averages_op)
        self.each_sample_loss = tf.concat(each_sample_loss_list, axis=0)
        self.predictions = tf.concat(tower_predictions, axis=0)

    def predict(self, x, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.x: x}
        return sess.run([self.predictions], feed_dict=feed_dict)

    def update(self, x, y_, lr, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.x: x, self.y_: y_, self.lr: lr}
        _, loss = sess.run([self.train_op, self.batch_mean_loss], feed_dict=feed_dict)
        return loss

    def get_accuracy(self, x, y_, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.x: x, self.y_: y_}
        return sess.run([self.accuracy], feed_dict=feed_dict)


if __name__ == '__main__':
    mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    model_1 = Classifier(global_step)
    init_op = tf.initialize_all_variables()
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init_op)
        learning_rate = FLAGS.learning_rate
        for i in range(10000):
            train_x, train_y = mnist.train.next_batch(30)
            test_x, test_y = mnist.test.next_batch(30)
            train_x = train_x.reshape(-1, 28, 28, 1)
            test_x = test_x.reshape(-1, 28, 28, 1)
            print("train_x shape is:{}".format(train_x.shape))
            cost = model_1.update(train_x, train_y, learning_rate, sess)
            prediction = model_1.predict(train_x, sess)
            accuracy = model_1.get_accuracy(test_x, test_y, sess)
            print("step:{}, cost:{}, accuracy:{}".format(i, cost, accuracy))
            decay_steps = int(1000)
            exp_term = math.floor(float(sess.run(global_step)) / decay_steps)
            learning_rate = FLAGS.learning_rate * math.pow(FLAGS.learning_rate_decay_factor, exp_term)
            print("learning_rate is:{}".format(learning_rate))
