"""

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.

DDPG is Actor Critic based algorithm.

Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:

tensorflow 1.0

gym 0.8.0

"""

import tensorflow as tf
import numpy as np
import time
import pandas as pd

from environmen import CITY





#####################  hyper parameters  ####################



MAX_EPISODES = 200
MAX_EP_STEPS1 = 1000
MAX_EP_STEPS2 = 1000

LR_A = 0.001    # learning rate for actor

LR_C = 0.001    # learning rate for critic

GAMMA = 0.9     # reward discount

TAU = 0.1      # soft replacement

MEMORY_CAPACITY = 2000

BATCH_SIZE = 32



RENDER = False

ENV_NAME = 'Pendulum-v0'



###############################  DDPG  ####################################



class DDPG(object):

    def __init__(self, a_dim, s_dim, a_bound):

        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)

        self.pointer = 0

        self.sess = tf.Session()
        self.mu = np.array([300, 390, 305, 305, 305, 365, 385, 361, 300, 308])
        # self.mu = 310 * np.ones(self.state_space)
        self.lambda_bar = 3.0



        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')

        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')

        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a_ = np.zeros([BATCH_SIZE, a_dim], dtype=np.float32)






        with tf.variable_scope('Actor'):

            self.a = self._build_a(self.S, scope='eval', trainable=True)



            #a_ = self._build_a(self.S_, scope='target', trainable=False)



        with tf.variable_scope('Critic'):

            # assign self.a = a in memory when calculating q for td_error,

            # otherwise the self.a is from Actor when updating Actor

            q = self._build_c(self.S, self.a, scope='eval', trainable=True)

            q_ = self._build_c(self.S_, self.a_, scope='target', trainable=False)



        # networks parameters

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')

        #self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')

        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')

        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')



        # target net replacement

        self.soft_replace = [[tf.assign(tc, (1 - TAU) * tc + TAU * ec)]

                             for tc, ec in zip( self.ct_params, self.ce_params)]



        q_target = self.R + GAMMA * q_

        # in the feed_dic for the td_error, the self.a should change to actions in memory

        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)

        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)



        a_loss = - tf.reduce_mean(q)    # maximize the q

        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)



        self.sess.run(tf.global_variables_initializer())

    def mf(self, s):
        S = self.s_dim
        #F = tf.placeholder(tf.float32, [None, s_dim])
        N = 100
        dt = 0.1
        F = (self.lambda_bar * self.mu) / pow((self.mu - (s * N * self.lambda_bar)), 2) + 0.2 * self.lambda_bar * pow(self.mu , 2)/(2 * pow(10, 7))
        #F = (self.lambda_bar * self.mu) / pow((self.mu - (s * N * self.lambda_bar)), 2)

        A = np.zeros((BATCH_SIZE, S, S))
        A[:, 0, 0] = - np.maximum(F[:, 0] - F[:, 1], 0)
        A[:, 0, 1] = np.maximum(F[:, 1] - F[:, 0], 0)

        A[:, S - 1, S - 1] = - np.maximum(F[:, S - 1] - F[:, S - 2], 0)
        A[:, S - 1, S - 2] = np.maximum(F[:, S - 2] - F[:, S - 1], 0)

        for i in range(1, S - 1):
            A[:, i, i] = -np.maximum(F[:, i] - F[:, i + 1], 0) - np.maximum(F[:, i] - F[:, i - 1], 0)
            A[:, i, i + 1] = np.maximum(F[:, i + 1] - F[:, i], 0)
            A[:, i, i - 1] = np.maximum(F[:, i - 1] - F[:, i], 0)

        A = A * dt
        a = np.zeros([BATCH_SIZE,S - 1])
        for k in range(BATCH_SIZE):
            for i in range(S):
                for j in range(S):
                    if (i != j) and (A[k, i, j] != 0):
                        if i < j:
                            a[k, i] = - A[k, i, j]
                        else:
                            a[k, j] = A[k, i, j]
        # print(a)
        return a





    def choose_action(self, s):

        res = self.sess.run(self.a, {self.S: s[np.newaxis, :]})

        chosen_a = np.reshape(res, [self.a_dim, ])
        # print(chosen_a)
        return chosen_a
        #return self.sess.run(self.a, {self.S: s[None, :]})[0, :]



    def learn(self):

        # soft target replacement

        self.sess.run(self.soft_replace)




        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)

        bt = self.memory[indices, :]

        bs = bt[:, :self.s_dim]

        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]

        br = bt[:, -self.s_dim - 1: -self.s_dim]

        bs_ = bt[:, -self.s_dim:]

        self.a_ = self.mf(bs_)



        self.sess.run(self.atrain, {self.S: bs})

        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})



    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, a, [r], s_))

        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory

        self.memory[index, :] = transition

        self.pointer += 1



    def _build_a(self, s, scope, trainable):

        with tf.variable_scope(scope):

            net = tf.layers.dense(s, 128, activation=tf.nn.relu, name='l1', trainable=trainable)

            net1 = tf.layers.dense(net, 64, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)

            a = tf.layers.dense(net1, self.a_dim, activation=tf.nn.tanh, use_bias=False, name='a', trainable=trainable)
            out_a = tf.multiply(a, self.a_bound, name='scaled_a')

            return out_a


    def _build_c(self, s, a, scope, trainable):

        with tf.variable_scope(scope):

            n_l1 = 128

            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)

            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)

            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            net1 = tf.layers.dense(net, 64, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)

            return tf.layers.dense(net1, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################




s_dim = 10

a_dim = 9

a_bound = 0.5
ddpg = DDPG(a_dim, s_dim, a_bound)

var = 0.1  # control exploration

t1 = time.time()

Y = np.zeros(MAX_EP_STEPS1 + MAX_EP_STEPS2 * (MAX_EPISODES - 1))
Z = np.zeros(MAX_EP_STEPS1 + MAX_EP_STEPS2 * (MAX_EPISODES - 1))

for i in range(MAX_EPISODES):

    env = CITY()

    var = 0.1
    s, init_sum = env.reset()
    sum1 = 0

    if i == 0:
        for j in range(MAX_EP_STEPS1):

            if j % 1000 == 0:
                print(j)

            # Add exploration noise

            a = ddpg.choose_action(s)

            # print(a)
            a = np.random.normal(a, var)  # add randomness to action selection for exploration
            for k in range(a_dim):
                if a[k] > 0.5:
                    a[k] = 0.5
                elif a[k] < -0.5:
                    a[k] = -0.5

            # s_, r, done = env.step(a)

            s_, r, r1, r2 = env.step(a)
            # print(r)

            ddpg.store_transition(s, a, r, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness


                ddpg.learn()

            if j >= 900:
                sum1 += r1

            if j == 999:
                print(sum1/100)



            s = s_
            #print(-r2)

            # print(s)
            Y[j] = r1
            Z[j] = r2

    else:
        for j in range(MAX_EP_STEPS2):

            if j % 1000 == 0:
                print(j + (i-1) * MAX_EP_STEPS2 + MAX_EP_STEPS1)

            # Add exploration noise

            a = ddpg.choose_action(s)

            # print(a)
            a = np.random.normal(a, var)  # add randomness to action selection for exploration
            for k in range(a_dim):
                if a[k] > 0.5:
                    a[k] = 0.5
                elif a[k] < -0.5:
                    a[k] = -0.5

            # s_, r, done = env.step(a)

            s_, r, r1, r2 = env.step(a)
            # print(r)

            ddpg.store_transition(s, a, r, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness

                ddpg.learn()

            if j >= 900:
                sum1 += r1

            if j == 999:
                print(sum1 / 100)

            s = s_
            #print(-r)
            # print(s)
            Y[MAX_EP_STEPS1 + j + MAX_EP_STEPS2 * (i-1)] = r1
            Z[MAX_EP_STEPS1 + j + MAX_EP_STEPS2 * (i - 1)] = r2













print('Running time: ', time.time() - t1)

pd_data2 = pd.DataFrame(Y)
s2 = 'newmfrl_time{0}'.format(1)
pd_data2.to_csv(s2)

pd_data3 = pd.DataFrame(Z)
s3 = 'newmfrl_energy{0}'.format(1)
pd_data3.to_csv(s3)



