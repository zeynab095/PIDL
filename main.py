import tensorflow as tf
from PCA_eq_C import start_train
#from synthetic_data import visualize_data, get_data, do_timestep
from k_synt_data import do_timestep, visualize_data, get_data
import numpy as np
import tensorflow as tf

w = h = 2
dx = dy = 0.01
nx, ny = int(w / dx), int(h / dy)
u0 = np.zeros((nx, ny))
u = u0.copy()
Cb0 = np.zeros((nx, ny))
Cb = Cb0.copy()
t0 = 0
time_steps = 580
time_interval = 20

#get_data(time_steps, time_interval, u0, u, t0, Cb0, Cb, True)
start_train()
# import matplotlib.pyplot as plt
# loss_values = [0.69411586222116872, 0.6923803442491846, 0.66657293575365906, 0.43212054205535255, 0.23119813830216157]
# loss_values1 = [2.0,1.5,2.45,2.67,2.99]
# plt.plot(loss_values, label="1")
# plt.plot(loss_values1, label="2")
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
#
# plt.show()

import os

#print(os.listdir('models/'))
