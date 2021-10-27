# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:43:53 2021

@author: zeyna
"""


from k_synt_data import get_data
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import time
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from util import neural_net, neural_net_k, Navier_Stokes_2D, PCA_2D, \
    tf_session, mean_squared_error, relative_error

class HFM(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _star: preditions

    def __init__(self, t_data, x_data, y_data, Ci_data, Cb_data,
                 t_eqns, x_eqns, y_eqns,
                 layers, batch_size,
                 kon, koff, R0, D, Lv, Cv, SV):

        # specs
        self.layers = layers
        self.batch_size = batch_size

        # flow properties
        self.kon = kon
        self.koff = koff
        self.R0 = R0
        self.D = D
        self.Lv = Lv
        self.Cv = Cv
        self.SV = SV
        # data
        [self.t_data, self.x_data, self.y_data, self.Ci_data, self.Cb_data, self.kon_data, self.koff_data] = [t_data, x_data, y_data, Ci_data, Cb_data, kon, koff]
        [self.t_eqns, self.x_eqns, self.y_eqns] = [t_eqns, x_eqns, y_eqns]

        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.Ci_data_tf, self.Cb_data_tf, self.koff_data_tf, self.kon_data_tf] = [
            tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(7)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf] = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _
                                                            in range(3)]


        self.net_cuvp = neural_net(self.t_data, self.x_data, self.y_data,
                                   layers=self.layers)

        [self.Ci_data_pred, self.kon_data_pred, self.koff_data_pred] = self.net_cuvp(self.t_data_tf,
                                                                    self.x_data_tf,
                                                                    self.y_data_tf)
        print("ci tf shape: ", self.Ci_data_tf.shape)
        print("cb tf shape: ", self.Cb_data_tf.shape)
        [self.e1_eqns_pred,
         self.e2_eqns_pred, self.e3_eqns_pred] = PCA_2D(self.Ci_data_tf,
                                     self.Cb_data_tf,
                                     self.t_data_tf,
                                     self.x_data_tf,
                                     self.y_data_tf,
                                     self.kon_data_pred,
                                     self.koff_data_pred,
                                     self.R0,
                                     self.D,
                                     self.Lv,
                                     self.Cv,
                                     self.SV)


        self.loss = mean_squared_error(self.kon_data_pred, self.kon_data_tf) + \
                    mean_squared_error(self.koff_data_pred, self.koff_data_tf) + \
                    mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    mean_squared_error(self.e2_eqns_pred, 0.0)



            # optimizers
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)


        self.sess = tf_session()

    def train(self, total_time, learning_rate):

        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]
        saver = tf.train.Saver()

        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:

            idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
            idx_eqns = np.random.choice(N_eqns, self.batch_size)
            # print("idx_data: ", idx_data)
            # print("idx_eqns: ", idx_eqns)

            (t_data_batch,
             x_data_batch,
             y_data_batch,
             Ci_data_batch,
             Cb_data_batch,
             kon_data_batch,
             koff_data_batch) = (self.t_data[idx_data, :],
                               self.x_data[idx_data, :],
                               self.y_data[idx_data, :],
                               self.Ci_data[idx_data, :],
                               self.Cb_data[idx_data, :],
                               self.kon_data[idx_data, :],
                               self.koff_data[idx_data, :])

            (t_eqns_batch,
             x_eqns_batch,
             y_eqns_batch) = (self.t_eqns[idx_eqns, :],
                              self.x_eqns[idx_eqns, :],
                              self.y_eqns[idx_eqns, :])

            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.Ci_data_tf: Ci_data_batch,
                       self.Cb_data_tf: Cb_data_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.kon_data_tf: kon_data_batch,
                       self.koff_data_tf: koff_data_batch,
                       self.learning_rate: learning_rate}

            self.sess.run([self.train_op], tf_dict)
            if it % 1000 == 0:
                saver.save(self.sess, "models/model_eq_C_08")
                print("Saving model.............................")

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed / 3600.0
                [loss_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.learning_rate], tf_dict)

                print('Pcae_eq_C It: %d, Loss all: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      % (it, loss_value,  elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
            it += 1


    def predict(self, t_star, x_star, y_star, kon_star, koff_star):

        # tf_dict_k = {self.x_data_tf: x_star, self.y_data_tf: y_star}
        # kon_star = self.sess.run(self.kon_data_pred, tf_dict_k)
        # koff_star = self.sess.run(self.koff_data_pred, tf_dict_k)

        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star, self.kon_data_tf: kon_star,
                   self.koff_data_tf: koff_star}

        kon_star = self.sess.run(self.kon_data_pred, tf_dict)
        koff_star = self.sess.run(self.koff_data_pred, tf_dict)
        #Cb_star = self.sess.run(self.Cb_data_pred, tf_dict)

        return kon_star, koff_star



def start_train():
    batch_size = 40000

    layers = [3] + 20 * [4 * 50] + [3]
    Tcool, Thot = 0, 1
    w = h = 2
    dx = dy = 0.01
    nx, ny = int(w / dx), int(h / dy)
    u0 = np.zeros((nx, ny))
    u = u0.copy()
    Cb0 = np.zeros((nx, ny))
    Cb = Cb0.copy()
    t0 = 0
    time_steps = 580
    time_interval = 1
    train_n = int(time_steps/time_interval)
    np_input_array, num_items, kon_data, koff_data = get_data(time_steps, time_interval, u0, u, t0, Cb0, Cb)

    nx = 200
    ny = 200
    center_x = nx / 2
    center_y = ny / 2
    radius = 50
    y, x = np.ogrid[-center_x:nx - center_x, -center_y:ny - center_y]
    mask = x * x + y * y <= radius * radius

    kon = np.full((nx, ny), 7.7 * 10 ** -1)
    kon[mask] = 7.7

    koff = np.full((nx, ny), 7.7 * 10 ** -4)
    koff[mask] = 7.7 * 10 ** -3

    kon = kon.reshape(40000, 1)
    koff = koff.reshape(40000, 1)

    kon_train = np.tile(kon, train_n).flatten(order='F').reshape(num_items, 1)
    koff_train = np.tile(koff, train_n).flatten(order='F').reshape(num_items, 1)


    print("shape: ", np_input_array.shape)
    print("x shape: ", np_input_array[:, 0].shape)
    print("y shape: ", np_input_array[:, 1].shape)
    print("t shape: ", np_input_array[:, 2].shape)
    print("Ci shape: ", np_input_array[:, 3].shape)
    print("Cb shape: ", np_input_array[:, 4].shape)
    print("kon shape: ", kon_train.shape)
    print("koff shape: ", koff_train.shape)

    x_data = np_input_array[:, 0].reshape(num_items, 1)
    y_data = np_input_array[:, 1].reshape(num_items, 1)
    t_data = np_input_array[:, 2].reshape(num_items, 1)
    Ci_data = np_input_array[:, 3].reshape(num_items, 1)
    Cb_data = np_input_array[:, 4].reshape(num_items, 1)

    x_eqns = np_input_array[:, 0].reshape(num_items, 1)
    y_eqns = np_input_array[:, 1].reshape(num_items, 1)
    t_eqns = np_input_array[:, 2].reshape(num_items, 1)


    #
    D = 8.7 * 10 ** -5
    Cv = 1
    Lv = 3.3 * 10 ** -2
    # R0 = np.random.normal(loc=4.089 * 10**-1, scale=8 * 10**-2, size=(nx, ny)).reshape(40000,1)
    R0 = 4.089 * 10 ** -1
    SV = 0.35


        # Training
    model = HFM(t_data, x_data, y_data, Ci_data, Cb_data,
                t_eqns, x_eqns, y_eqns,
                layers, batch_size,
                kon_train, koff_train, R0, D, Lv, Cv, SV)
    print("--------------------------------------------------------------------------------")
    #model.restore()
    print("--------------------------------------------------------------------------------")
    model.train(total_time=0.01, learning_rate=1e-3)
    # =============================================================================
    time_steps_test = 600
    time_interval_test = 50

    np_input_array_test, num_items_test, kon_data_test, koff_data_test = get_data(time_steps_test, time_interval_test, u0, u, t0, Cb0, Cb)
    print(num_items_test)

    x_test = np_input_array_test[:, 0].reshape(num_items_test, 1)
    y_test = np_input_array_test[:, 1].reshape(num_items_test, 1)
    t_test = np_input_array_test[:, 2].reshape(num_items_test, 1)
    Ci_test = np_input_array_test[:, 3].reshape(num_items_test, 1)
    Cb_test = np_input_array_test[:, 4].reshape(num_items_test, 1)


    test_n = int(time_steps_test / time_interval_test)
    kon_test = np.tile(kon, test_n).flatten(order='F').reshape(num_items_test, 1)
    koff_test = np.tile(koff, test_n).flatten(order='F').reshape(num_items_test, 1)
    # Prediction
    Ci_pred = model.predict(t_test, x_test, y_test, kon_test, koff_test)
    print(Ci_pred)


