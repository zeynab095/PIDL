import tensorflow as tf

from PCA_kall_eq import start_train
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

#do_timestep(u0, u, t0, Cb0, Cb)
#visualize_data(time_steps, u0, u, t0, Cb0, Cb)
#get_data(time_steps, time_interval, u0, u, t0, Cb0, Cb, True)
start_train()
#
# #Prepare to feed input, i.e. feed_dict and placeholders
# w1 = tf.placeholder("float", name="w1")
# w2 = tf.placeholder("float", name="w2")
# b1= tf.Variable(2.0,name="bias")
# feed_dict ={w1:4,w2:8}
#
# #Define a test operation that we will restore
# w3 = tf.add(w1,w2)
# w4 = tf.multiply(w3,b1,name="op_to_restore")
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# #Create a saver object which will save all the variables
# saver = tf.compat.v1.train.Saver()
#
# #Run the operation by feeding input
# print(sess.run(w4,feed_dict))
# #Prints 24 which is sum of (w1+w2)*b1
#
# #Now, save the graph
# saver.save(sess, 'my_test_model')
#
#
# sess=tf.Session()
# #First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph('my-model212.meta')
# saver.restore(sess,tf.train.latest_checkpoint('./'))
#
#
# # Access saved Variables directly
# print(sess.run())
# # This will print 2, which is the value of bias that we saved
#
#
# # Now, let's access and create placeholders variables and
# # create feed-dict to feed new data
#
# graph = tf.get_default_graph()
# w1 = graph.get_tensor_by_name("w1:0")
# w2 = graph.get_tensor_by_name("w2:0")
# feed_dict ={w1:13.0,w2:17.0}
#
# #Now, access the op that you want to run.
# op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
#
# print(sess.run(op_to_restore,feed_dict))