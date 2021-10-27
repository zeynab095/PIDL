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

start_train()

import os

#print(os.listdir('models/'))
