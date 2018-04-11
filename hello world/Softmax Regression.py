#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)#載入數據

sess=tf.InteractiveSession()                              #創建一個新的session
x=tf.placeholder(tf.float32,[None,784])                   #輸入數據的地方

w=tf.Variable(tf.zeros([784,10]))                         #權值矩陣
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,w)+b)                         #Soft Regression算法實現

#Training
y_=tf.placeholder(tf.float32,[None,10])                   #輸入數據的地方
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))#損失函數indices=[1]指第二维求和
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)       #指定優化器
#Run
tf.global_variables_initializer().run()                   #全局參數初始化器

for i in range(1000):                                     #迭代地完成訓練
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})

#evaluate
correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))