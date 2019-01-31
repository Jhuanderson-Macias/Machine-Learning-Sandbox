#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 22:40:50 2019

@author: jhuandersonmacias
"""

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from sklearn.preprocessing import scale


X,Y = load_boston(True)

X,Y = shuffle(X,Y)

X_train = scale(X[:300])
Y_train =  Y[:300]

X_test = scale(X[300:])
Y_test =  Y[:300]

W = tf.Variable(tf.random.normal([13,1], mean=0, stddev=1.0, dtype=tf.float64))
b = tf.Variable(tf.zeros(1, dtype=tf.float64))


x=tf.placeholder(tf.float64)
y=tf.placeholder(tf.float64)

# we have a function y=wx + b

pred = tf.add(b, tf.matmul(x,W))
squared_delta = tf.square(y-pred) # R^2

loss = tf.reduce_mean(squared_delta)

# these varibles are declared at run time 

init = tf.global_variables_initializer()

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

cost_history =  []
epochs = 5000

with tf.Session() as sess:
    sess.run(init)
    #evalute a tensor flow session. This will initilize our W and B variables
    
    for i in list(range(epochs)):
        sess.run(optimizer, {x: X_train, y: Y_train})
    
        if i % 10 ==0:
            cost_history.append(sess.run(loss,  {x: X_train, y: Y_train}))
        if i % 500 == 0:
            print(sess.run(loss,  {x: X_train, y: Y_train}))
        
    plt.plot(cost_history)
    plt.show()
    
    print("error on test data",  sess.run(loss,  {x: X_test, y: Y_test}))
    
    sess.close()
    
        
    
        
        
 



