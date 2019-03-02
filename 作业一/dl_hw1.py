#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:54:19 2019

@author: sunyirong
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('./MNIST-data',
                                  one_hot=True)

BATCH_SIZE = 200
STEPS=10000
#xlabel=[]


x=tf.placeholder(tf.float32,shape=(None,784))
y_=tf.placeholder(tf.float32,shape=(None,10))

w1=tf.Variable(tf.random_normal([784,256],stddev=1))
b1=tf.Variable(tf.random_normal([256],stddev=1))
z1=tf.matmul(x,w1)+b1
a1=tf.nn.relu(z1)

w2=tf.Variable(tf.random_normal([256,10],stddev=1))
b2=tf.Variable(tf.random_normal([10],stddev=1))
#y=tf.nn.softmax(tf.matmul(a1,w2)+b2)
y=tf.matmul(a1,w2)+b2

#loss = tf.reduce_mean(tf.square(y-y_))
#right_num=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#loss=tf.reduce_mean(tf.cast(right_num,tf.float32))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
loss=tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


right_num=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(right_num,tf.float32))

xlabel=[]
ylabel=[]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(STEPS):
        xs,ys=mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step,feed_dict={x:xs,y_:ys})
        if i%100 ==0:
            total_accuracy = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
            xlabel.append(i)
            ylabel.append(1-total_accuracy)
            print("After %d training steps,loss on all data is %g"%(i,1-total_accuracy))
            
plt.plot(xlabel,ylabel)
plt.show()

            
         
            
            
            
            
            

            
            
        

