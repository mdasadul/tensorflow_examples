from __future__ import print_function

import numpy as np
import pandas as pd 
import sklearn.datasets as datasets
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf 


x = tf.placeholder(tf.float32,[None,None],name="x")
y = tf.placeholder(tf.float32,[None,None],name="y")
x_ = np.random.rand(500,2)
x_[:,1] = 2*x_[:,0]+np.random.rand(500)
dataframe = pd.DataFrame(x_,columns=['x','y'])


x_values=dataframe['x'].values[:,np.newaxis]
y_values=dataframe['y'].values[:,np.newaxis]

with tf.variable_scope('wb') as wb:
    w = tf.get_variable('w',[1])
    b= tf.get_variable('b',[1])

def logistic_regression():
    pred = w*x+b
    loss = tf.losses.mean_squared_error(pred,y)
    return pred,loss


def run():
    pred, loss = logistic_regression()
    train_loss = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()
    feed_dict = {x:x_values,y:y_values}
    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            loss_ ,_= sess.run([loss,train_loss],feed_dict)
            if i %500 ==0:
                print(loss_)
        pred_ = sess.run(pred,feed_dict)
    return pred_

pred_ = run()
plt.scatter(x_values,y_values,c='g')
plt.plot(x_values,pred_)
plt.show()