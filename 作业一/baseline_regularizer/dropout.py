#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:54:19 2019

@author: sunyirong
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('../MNIST-data', one_hot=True)

BATCH_SIZE = 32
STEPS=mnist.train.num_examples//BATCH_SIZE
epochs = 50


x=tf.placeholder(tf.float32,shape=(None,784))
y_=tf.placeholder(tf.float32,shape=(None,10))

w1=tf.Variable(tf.random_normal([784,256],stddev=0.01))
b1=tf.Variable(tf.random_normal([256],stddev=0.01))
z1=tf.matmul(x,w1)+b1
a1=tf.nn.relu(z1)
#tf.add_to_collection(tf.GraphKeys.WEIGHTS,w1)

#添加dropout正则化
a2 = tf.nn.dropout(a1,0.5)

#w2=tf.Variable(tf.random_normal([256,128],stddev=1))
#b2=tf.Variable(tf.random_normal([128],stddev=1))
#z2=tf.matmul(a1,w2)+b2
#a2=tf.nn.relu(z2)

w3 = tf.Variable(tf.random_normal([256,10],stddev=0.01))
b3 = tf.Variable(tf.random_normal([10],stddev=0.01))
y = tf.matmul(a2,w3)+b3
#tf.add_to_collection(tf.GraphKeys.WEIGHTS,w3)



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
loss=tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss)



right_num=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(right_num,tf.float32))

xlabel=[]
loss_label=[]
acc_label=[]


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(1,epochs+1):
        for j in range(STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step,feed_dict={x:xs,y_:ys})
#            if j%200 ==0:
#                total_loss = sess.run(loss,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
#                print('    After %d steps,loss on all data is %g'%(j,total_loss))
#                if j!=0:
#                    xlabel.append(j+i*10)
#                    ylabel.append(total_loss)
        total_loss, total_acc= sess.run((loss,accuracy),feed_dict={x:mnist.train.images,y_:mnist.train.labels})
        xlabel.append(i+1)
        loss_label.append(total_loss)
        acc_label.append(total_acc)
        print("After %d epochs,loss on all data is %g,and the accuracy is %g"%(i,total_loss,total_acc))
      
plt.plot(xlabel,loss_label)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

            
         
            
            
            
            
            

            
            
        


