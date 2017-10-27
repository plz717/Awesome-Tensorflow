# THis scipt is to identify the effectiveness of actor algorithm used in rl_noisy data
from __future__ import print_function

import tensorflow as tf
import gym
import numpy as np
import random
np.set_printoptions(threshold=np.nan)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('log_dir', '.', 'dir to store operations summary')


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name)


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name)


class REINFORCE():
    """objecttive function is log pi(a|s) * r_t
         input:
             s:[None, states_dim]
             action:[None, actions_dim]
             target:tf.float32
    """

    def __init__(self, scope="policy_estimor_REINFORCE"):

        self.s = tf.placeholder(tf.float32, [None, 4])
        self.action = tf.placeholder(tf.float32, [None, 2])
        self.target = tf.placeholder(tf.float32, name="target")

        w1 = weight_variable([4, 2], "w1")
        b1 = bias_variable([2], "b1")
        readout = tf.matmul(self.s, w1) + b1
        target_float = tf.cast(self.target, tf.float32)
        self.loss = target_float * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.action, logits=readout))

        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        self.action_probs = tf.nn.softmax(readout)
        self.summary_op = tf.summary.scalar('loss', self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.s: state}
        return sess.run(self.action_probs, feed_dict=feed_dict)

    def update(self, state, action, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.s: state, self.action: action, self.target: target}
        _, loss, summary_op = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary_op


def MakeSummary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary


def main():
    actor = REINFORCE()

    env = gym.make('CartPole-v0')
    # env.monitor.start('cartpole-hill/', force=True)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)

        sess.run(tf.initialize_all_variables())
        observation = env.reset()
        for i in xrange(20000):
            observation = env.reset()
            totalreward = 0
            transitions = []
            for _ in xrange(200):
                # calculate policy
                s_t = np.expand_dims(observation, axis=0)
                probs = actor.predict(s_t)
                action = 0 if random.uniform(0, 1) < probs[0][0] else 1
                # record the transition
                a_t = np.zeros((1, 2))
                a_t[0][action] = 1
                #print("state:{},action:{}".format(s_t, a_t))
                # take the action in the environment
                observation, r_t, done, info = env.step(action)
                transitions.append((s_t, a_t, r_t))

                totalreward += r_t

                if done:
                    print("totalreward is:", totalreward)
                    train_writer.add_summary(MakeSummary("episode_reward", totalreward), i * 200 + _)
                    break
            for index, trans in enumerate(transitions):
                s_t, a_t, r_t = trans
                v_t = sum(FLAGS.dis_factor**i * item[2] for i, item in enumerate(transitions[index:]))

                loss, actor_aummary_op = actor.update(s_t, a_t, v_t, sess=sess)
                #print("loss is:", loss)
                train_writer.add_summary(actor_aummary_op, i * 200 + index)

main()
