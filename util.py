"""
@author: Maziar Raissi
"""

import tensorflow as tf
import numpy as np


def tf_session():
    # tf session
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.compat.v1.Session(config=config)

    # init
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    return sess


def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact)) / np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact)) / tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))


def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))

def mean_abs_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.abs(pred - exact))


def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy)[0]
    Y_x = tf.gradients(G, dummy)[0]
    return Y_x



class neural_net(object):
    def __init__(self, *inputs, layers):

        self.layers = layers
        self.num_layers = len(self.layers)

        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)

        self.weights = []
        self.biases = []
        self.gammas = []

        for l in range(0, self.num_layers - 1):
            in_dim = self.layers[l]
            out_dim = self.layers[l + 1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))

    def __call__(self, *inputs):

        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std

        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W / tf.norm(W, axis=0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g * H + b
            # activation
            if l < self.num_layers - 2:
                H = H * tf.sigmoid(H)

        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
        #Y= Y*[1.0, 0.01]
        print("from neural net: ", len(Y))
        print("y from neural net: ", Y)
        #multiple
        return Y

# in util.py
class neural_net_special(object):

    # def __init__(self, in_x, in_y, in_t, layers):
    def __init__(self, *inputs, layers):

        self.layers = [3] + layers[1:-1] + [4]  # a fixed architecture, inputs are x,y,t, outputs are cicb, konkoff
        self.num_layers = len(self.layers)

        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)

        self.weights = []
        self.biases = []
        self.gammas = []
        # self.weights[:self.num_layers - 1] weights for the first net,
        # self.weights[self.num_layers - 1:] weights for the second net

        for i in range(2):  # we have two sub-nets
            # two sub-nets have same architecture only differs in the input_dim, 2 for xy-sub-net, 1 for t-sub-net
            sub_layers = [2 if i == 0 else 1] + self.layers[1:-1] + [2]
            name_scope = 'KonKoff' if i == 0 else 'CiCb'
            with tf.variable_scope(name_scope):  # this is to separate the name for kon, koff and ci, cb
                for l in range(0, self.num_layers - 1):
                    in_dim = sub_layers[l]
                    out_dim = sub_layers[l + 1]

                    if i > 0 and l == 1:  # to concat features
                        in_dim += sub_layers[-2]

                    W = np.random.normal(size=[in_dim, out_dim])
                    b = np.zeros([1, out_dim])
                    g = np.ones([1, out_dim])
                    # tensorflow variables
                    self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
                    self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
                    self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))

    def __call__(self, *inputs):

        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std
        H_xy, H_t = tf.split(H, [2, 1], axis=1)

        H = H_xy

        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W / tf.norm(W, axis=0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g * H + b
            # activation
            if l < self.num_layers - 2:
                H = H * tf.sigmoid(H)
            # save the feature
            if l == self.num_layers - 3:
                xy_feature = H

        xy_output = H
        H = H_t

        for l in range(0, self.num_layers - 1):
            W = self.weights[l + self.num_layers - 1]
            b = self.biases[l + self.num_layers - 1]
            g = self.gammas[l + self.num_layers - 1]
            # weight normalization
            V = W / tf.norm(W, axis=0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g * H + b
            # activation
            if l < self.num_layers - 2:
                H = H * tf.sigmoid(H)
            # read the feature
            if l == 0:
                H = tf.concat([xy_feature, H], axis=1)

        xyt_output = H
        H = tf.concat([xy_output, xyt_output], axis=1)

        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
        # Y= Y*[1.0, 0.01]
        # here, I think you still need to apply a scaling
        print("from neural net: ", len(Y))
        print("y from neural net: ", Y)
        # multiple
        return Y

class neural_net_k(object):
    def __init__(self, *inputs, layers):

        self.layers = layers
        self.num_layers = len(self.layers)

        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)

        self.weights = []
        self.biases = []
        self.gammas = []

        for l in range(0, self.num_layers - 1):
            in_dim = self.layers[l]
            out_dim = self.layers[l + 1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))

    def __call__(self, *inputs):

        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std

        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W / tf.norm(W, axis=0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g * H + b
            # activation
            if l < self.num_layers - 2:
                H = H * tf.sigmoid(H)

        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
        # Y= Y*[1.0, 0.01]
        Y[1] = Y[1] * 0.001
        print("from neural net: ", len(Y))
        print("y from neural net: ", Y)
        # multiple
        return Y


class neural_net1(object):
    def __init__(self, *inputs, layers):

        self.layers = layers
        self.num_layers = len(self.layers)

        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)

        self.weights = []
        self.biases = []
        self.gammas = []

        for l in range(0, self.num_layers - 1):
            in_dim = self.layers[l]
            out_dim = self.layers[l + 1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))

    def __call__(self, *inputs):

        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std

        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W / tf.norm(W, axis=0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g * H + b
            # activation
            if l < self.num_layers - 2:
                H = H * tf.sigmoid(H)

        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)

        return Y


def Navier_Stokes_2D(c, u, v, p, t, x, y, Pec, Rey):
    Y = tf.concat([c, u, v, p], 1)

    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)

    c = Y[:, 0:1]
    u = Y[:, 1:2]
    v = Y[:, 2:3]
    p = Y[:, 3:4]

    c_t = Y_t[:, 0:1]
    u_t = Y_t[:, 1:2]
    v_t = Y_t[:, 2:3]

    c_x = Y_x[:, 0:1]
    u_x = Y_x[:, 1:2]
    v_x = Y_x[:, 2:3]
    p_x = Y_x[:, 3:4]

    c_y = Y_y[:, 0:1]
    u_y = Y_y[:, 1:2]
    v_y = Y_y[:, 2:3]
    p_y = Y_y[:, 3:4]

    c_xx = Y_xx[:, 0:1]
    u_xx = Y_xx[:, 1:2]
    v_xx = Y_xx[:, 2:3]

    c_yy = Y_yy[:, 0:1]
    u_yy = Y_yy[:, 1:2]
    v_yy = Y_yy[:, 2:3]

    e1 = c_t + (u * c_x + v * c_y) - (1.0 / Pec) * (c_xx + c_yy)
    e2 = u_t + (u * u_x + v * u_y) + p_x - (1.0 / Rey) * (u_xx + u_yy)
    e3 = v_t + (u * v_x + v * v_y) + p_y - (1.0 / Rey) * (v_xx + v_yy)
    e4 = u_x + v_y

    return e1, e2, e3, e4


def PCA_2D(Ci, Cb, t, x, y, kon, koff, R0, D, Lv, Cv, SV):
    Y = tf.concat([Ci, Cb], 1)

    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)

    # =============================================================================
    Ci = Y[:, 0:1]
    Cb = Y[:, 1:2]

    Ci_t = Y_t[:, 0:1]
    Cb_t = Y_t[:, 1:2]

    Ci_x = Y_x[:, 0:1]
    Cb_x = Y_x[:, 1:2]

    Ci_y = Y_y[:, 0:1]
    Cb_y = Y_y[:, 1:2]

    Ci_xx = Y_xx[:, 0:1]
    Cb_xx = Y_xx[:, 1:2]

    Ci_yy = Y_yy[:, 0:1]
    Cb_yy = Y_yy[:, 1:2]
    # =============================================================================

    # =============================================================================
    e1 = koff * Cb + D * (Ci_xx + Ci_yy) - Ci * Ci_t - kon * Ci * (R0 - Cb) + R0 * (Ci_x + Ci_y) + Lv * (Cv - Ci) * SV
    e2 = kon * Ci * (R0 - Cb) - Cb * Cb_t - koff * Cb
    e3 = kon - 1000*koff
    # =============================================================================

    return e1, e2, e3


def PCA_special(kon, koff, Ci, Cb, x, y, t, R0, D, Lv, Cv, SV):
    # it is fine to use PCA_2D, but we actually don't need Cb_x, Cb_y or Cb_xx, Cb_yy, so this version is simplified.
    Y = tf.concat([Ci, Cb], 1)
    Y_t = fwd_gradients(Y, t)

    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)

    # =============================================================================
    Ci = Y[:, 0:1]
    Cb = Y[:, 1:2]

    Ci_t = Y_t[:, 0:1]
    Cb_t = Y_t[:, 1:2]

    Ci_x = fwd_gradients(Ci, x)
    Ci_y = fwd_gradients(Ci, y)
    Ci_xx = fwd_gradients(Ci_x, x)
    Ci_yy = fwd_gradients(Ci_y, y)
    # =============================================================================

    # =============================================================================
    e1 = koff * Cb + D * (Ci_xx + Ci_yy) - Ci * Ci_t - kon * Ci * (R0 - Cb) + R0 * (Ci_x + Ci_y) + Lv * (Cv - Ci) * SV
    e2 = kon * Ci * (R0 - Cb) - Cb * Cb_t - koff * Cb
    e3 = kon - 1000 * koff
    # =============================================================================

    return e1, e2, e3


def PCA_2D_1(kon, koff, t, x, y, Ci, Cb, R0, D, Lv, Cv, SV):
    Y = tf.concat([Ci, Cb], 1)

    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)

    # =============================================================================
    Ci = Y[:, 0:1]
    Cb = Y[:, 1:2]

    Ci_t = Y_t[:, 0:1]
    Cb_t = Y_t[:, 1:2]

    Ci_x = Y_x[:, 0:1]
    Cb_x = Y_x[:, 1:2]

    Ci_y = Y_y[:, 0:1]
    Cb_y = Y_y[:, 1:2]

    Ci_xx = Y_xx[:, 0:1]
    Cb_xx = Y_xx[:, 1:2]

    Ci_yy = Y_yy[:, 0:1]
    Cb_yy = Y_yy[:, 1:2]
    # =============================================================================

    # =============================================================================
    e1 = koff * Cb + D * (Ci_xx + Ci_yy) - Ci * Ci_t - kon * Ci * (R0 - Cb) + R0 * (Ci_x + Ci_y) + Lv * (Cv - Ci) * SV
    e2 = kon * Ci * (R0 - Cb) - Cb * Cb_t - koff * Cb
    e3 = kon - 1000*koff
    # =============================================================================

    return e1, e2, e3

def Gradient_Velocity_2D(u, v, x, y):
    Y = tf.concat([u, v], 1)

    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)

    u_x = Y_x[:, 0:1]
    v_x = Y_x[:, 1:2]

    u_y = Y_y[:, 0:1]
    v_y = Y_y[:, 1:2]

    return [u_x, v_x, u_y, v_y]


def Strain_Rate_2D(u, v, x, y):
    [u_x, v_x, u_y, v_y] = Gradient_Velocity_2D(u, v, x, y)

    eps11dot = u_x
    eps12dot = 0.5 * (v_x + u_y)
    eps22dot = v_y

    return [eps11dot, eps12dot, eps22dot]


def Navier_Stokes_3D(c, u, v, w, p, t, x, y, z, Pec, Rey):
    Y = tf.concat([c, u, v, w, p], 1)

    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    Y_zz = fwd_gradients(Y_z, z)

    c = Y[:, 0:1]
    u = Y[:, 1:2]
    v = Y[:, 2:3]
    w = Y[:, 3:4]
    p = Y[:, 4:5]

    c_t = Y_t[:, 0:1]
    u_t = Y_t[:, 1:2]
    v_t = Y_t[:, 2:3]
    w_t = Y_t[:, 3:4]

    c_x = Y_x[:, 0:1]
    u_x = Y_x[:, 1:2]
    v_x = Y_x[:, 2:3]
    w_x = Y_x[:, 3:4]
    p_x = Y_x[:, 4:5]

    c_y = Y_y[:, 0:1]
    u_y = Y_y[:, 1:2]
    v_y = Y_y[:, 2:3]
    w_y = Y_y[:, 3:4]
    p_y = Y_y[:, 4:5]

    c_z = Y_z[:, 0:1]
    u_z = Y_z[:, 1:2]
    v_z = Y_z[:, 2:3]
    w_z = Y_z[:, 3:4]
    p_z = Y_z[:, 4:5]

    c_xx = Y_xx[:, 0:1]
    u_xx = Y_xx[:, 1:2]
    v_xx = Y_xx[:, 2:3]
    w_xx = Y_xx[:, 3:4]

    c_yy = Y_yy[:, 0:1]
    u_yy = Y_yy[:, 1:2]
    v_yy = Y_yy[:, 2:3]
    w_yy = Y_yy[:, 3:4]

    c_zz = Y_zz[:, 0:1]
    u_zz = Y_zz[:, 1:2]
    v_zz = Y_zz[:, 2:3]
    w_zz = Y_zz[:, 3:4]

    e1 = c_t + (u * c_x + v * c_y + w * c_z) - (1.0 / Pec) * (c_xx + c_yy + c_zz)
    e2 = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1.0 / Rey) * (u_xx + u_yy + u_zz)
    e3 = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1.0 / Rey) * (v_xx + v_yy + v_zz)
    e4 = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1.0 / Rey) * (w_xx + w_yy + w_zz)
    e5 = u_x + v_y + w_z

    return e1, e2, e3, e4, e5


def Gradient_Velocity_3D(u, v, w, x, y, z):
    Y = tf.concat([u, v, w], 1)

    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)

    u_x = Y_x[:, 0:1]
    v_x = Y_x[:, 1:2]
    w_x = Y_x[:, 2:3]

    u_y = Y_y[:, 0:1]
    v_y = Y_y[:, 1:2]
    w_y = Y_y[:, 2:3]

    u_z = Y_z[:, 0:1]
    v_z = Y_z[:, 1:2]
    w_z = Y_z[:, 2:3]

    return [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z]


def Shear_Stress_3D(u, v, w, x, y, z, nx, ny, nz, Rey):
    [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z] = Gradient_Velocity_3D(u, v, w, x, y, z)

    uu = u_x + u_x
    uv = u_y + v_x
    uw = u_z + w_x
    vv = v_y + v_y
    vw = v_z + w_y
    ww = w_z + w_z

    sx = (uu * nx + uv * ny + uw * nz) / Rey
    sy = (uv * nx + vv * ny + vw * nz) / Rey
    sz = (uw * nx + vw * ny + ww * nz) / Rey

    return sx, sy, sz