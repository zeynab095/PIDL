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

from util import neural_net, Navier_Stokes_2D, PCA_2D, \
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
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.Ci_data_tf, self.Cb_data_tf, self.kon_data_tf, self.koff_data_tf] = [
            tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(7)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf] = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _
                                                            in range(3)]

        # physics "uninformed" neural networks
        self.net_cuvp = neural_net(self.t_data, self.x_data, self.y_data, layers=self.layers)
        print("net_cvup", self.net_cuvp)



        [self.Ci_data_pred, self.Cb_data_pred] = self.net_cuvp(self.t_data_tf,
                                                                    self.x_data_tf,
                                                                    self.y_data_tf)

        layersk = [2] + 10 * [4 * 10] + [2]
        self.net_cuvpk = neural_net(self.x_data, self.y_data, layers=layersk)
        [self.kon_data_pred, self.koff_data_pred] = self.net_cuvpk(self.x_data_tf,
                                                                     self.y_data_tf
                                                                    )

        # physics "informed" neural networks

        [self.Ci_eqns_pred,
         self.Cb_eqns_pred] = self.net_cuvp(self.t_eqns_tf,
                                                 self.x_eqns_tf,
                                                 self.y_eqns_tf)

        [self.e1_eqns_pred,
         self.e2_eqns_pred] = PCA_2D(self.Ci_eqns_pred,
                                     self.Cb_eqns_pred,
                                     self.t_eqns_tf,
                                     self.x_eqns_tf,
                                     self.y_eqns_tf,
                                     self.kon_data_pred,
                                     self.koff_data_pred,
                                     self.R0,
                                     self.D,
                                     self.Lv,
                                     self.Cv,
                                     self.SV)

        # loss
        # mean_squared_error(self.kon_data_pred, self.kon_data) + \
        # mean_squared_error(self.koff_data_pred, self.koff_data) + \
        self.loss = mean_squared_error(self.Ci_data_pred+self.Cb_data_pred, self.Ci_data_tf+self.Cb_data_tf) + \
                    mean_squared_error(self.kon_data_pred, self.kon_data) + \
                    mean_squared_error(self.koff_data_pred, self.koff_data) + \
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

        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:

            idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
            idx_eqns = np.random.choice(N_eqns, self.batch_size)

            (t_data_batch,
             x_data_batch,
             y_data_batch,
             Ci_data_batch,
             Cb_data_batch) = (self.t_data[idx_data, :],
                               self.x_data[idx_data, :],
                               self.y_data[idx_data, :],
                               self.Ci_data[idx_data, :],
                               self.Cb_data[idx_data, :])

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
                       self.learning_rate: learning_rate}

            self.sess.run([self.train_op], tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed / 3600.0
                [loss_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      % (it, loss_value, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
            it += 1

    def predict(self, t_star, x_star, y_star):

        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star}

        Ci_star = self.sess.run(self.Ci_data_pred, tf_dict)
        Cb_star = self.sess.run(self.Cb_data_pred, tf_dict)

        return Ci_star, Cb_star


def start_train():
    batch_size = 40000

    layers = [3] + 10 * [4 * 50] + [2]

    w = h = 2
    dx = dy = 0.01
    nx, ny = int(w / dx), int(h / dy)
    u0 = np.zeros((nx, ny))
    u = u0.copy()
    Cb0 = np.zeros((nx, ny))
    Cb = Cb0.copy()
    t0 = 0
    time_steps = 1000
    time_interval = 1
    np_input_array, num_items, kon_data, koff_data = get_data(time_steps, time_interval, u0, u, t0, Cb0, Cb)


    print("shape: ", np_input_array.shape)
    print("x shape: ", np_input_array[:, 0].shape)
    print("y shape: ", np_input_array[:, 1].shape)
    print("t shape: ", np_input_array[:, 2].shape)
    print("Ci shape: ", np_input_array[:, 3].shape)
    print("Cb shape: ", np_input_array[:, 4].shape)

    x_data = np_input_array[:, 0].reshape(num_items, 1)
    y_data = np_input_array[:, 1].reshape(num_items, 1)
    t_data = np_input_array[:, 2].reshape(num_items, 1)
    Ci_data = np_input_array[:, 3].reshape(num_items, 1)
    Cb_data = np_input_array[:, 4].reshape(num_items, 1)

    x_eqns = np_input_array[:, 0].reshape(num_items, 1)
    y_eqns = np_input_array[:, 1].reshape(num_items, 1)
    t_eqns = np_input_array[:, 2].reshape(num_items, 1)
    kon_data = kon_data.reshape(40000, 1)
    koff_data = kon_data.reshape(40000, 1)


      # row=1, col=0


    #
    #kon = 7.7 * 10 ** -1
    #koff = 7.7 * 10 ** -4

    # kon = np.full((40000, 1), 7.7 * 10 ** -1)
    # koff = np.full((40000, 1), 7.7 * 10 ** -4)
    nx = 200
    ny = 200
    center_x = nx / 2
    center_y = ny / 2
    radius = 50
    y, x = np.ogrid[-center_x:nx - center_x, -center_y:ny - center_y]
    mask = x * x + y * y <= radius * radius

    kon = np.full((nx, ny), 7.7 * 10 ** -1)
    kon[mask] = 15

    koff = np.full((nx, ny), 7.7 * 10 ** -4)
    koff[mask] = 15 * 10 ** -2

    kon = kon.reshape(40000, 1)
    koff = koff.reshape(40000, 1)

    D = 8.7 * 10 ** -5
    Cv = 1
    Lv = 3.3 * 10 ** -2
    # R0 = np.random.normal(loc=4.089 * 10**-1, scale=8 * 10**-2, size=(nx, ny)).reshape(40000,1)
    R0 = 4.089 * 10 ** -1
    SV = 0.35

    #
    #     # Training
    model = HFM(t_data, x_data, y_data, Ci_data, Cb_data,
                t_eqns, x_eqns, y_eqns,
                layers, batch_size,
                kon, koff, R0, D, Lv, Cv, SV)
    # =============================================================================
    model.train(total_time=0.2, learning_rate=1e-3)
    # =============================================================================
    time_steps_test = 1000
    time_interval_test = 50

    np_input_array_test, num_items_test, kon_data_test, koff_data_test = get_data(time_steps_test, time_interval_test, u0, u, t0, Cb0, Cb)
    print(num_items_test)

    x_test = np_input_array_test[:, 0].reshape(num_items_test, 1)
    y_test = np_input_array_test[:, 1].reshape(num_items_test, 1)
    t_test = np_input_array_test[:, 2].reshape(num_items_test, 1)
    Ci_test = np_input_array_test[:, 3].reshape(num_items_test, 1)
    Cb_test = np_input_array_test[:, 4].reshape(num_items_test, 1)

    #
    #     # Prediction
    Ci_pred, Cb_pred = model.predict(t_test, x_test, y_test)
    #
    #     # Error
    error_Ci = relative_error(Ci_pred, Ci_test)
    error_Cb = relative_error(Cb_pred, Cb_test)

    #
    print('Error Ci: %e' % (error_Ci))
    print('Error Cb: %e' % (error_Cb))

    error_Ci_mean = mean_squared_error(Ci_pred, Ci_test)
    error_Cb_mean = mean_squared_error(Cb_pred, Cb_test)

    #
    print('Error mean squared Ci: %e' % (error_Ci_mean))
    print('Error mean suqared Cb: %e' % (error_Cb_mean))

    Tcool, Thot = 0, 1

    print("Ci test shape: ", Ci_test.shape)
    for i in range(20):
        j = i*40000
        print(j)

        fig, ax = plt.subplots(2, 2)

        ax[0, 0].imshow(Ci_test[j:j+40000].reshape(200,200), cmap=plt.get_cmap('hot'),  vmin=Tcool, vmax=Thot)  # row=0, col=0
        ax[1, 0].imshow(Ci_pred[j:j+40000].reshape(200,200), cmap=plt.get_cmap('hot'),  vmin=Tcool, vmax=Thot)  # row=0,)  # row=1, col=0
        ax[0, 1].imshow(Cb_test[j:j + 40000].reshape(200, 200), cmap=plt.get_cmap('hot'),  vmin=Tcool, vmax=Thot)  # row=0, col=0
        ax[1, 1].imshow(Cb_pred[j:j + 40000].reshape(200, 200), cmap=plt.get_cmap('hot'),  vmin=Tcool, vmax=Thot)

        plt.show()

