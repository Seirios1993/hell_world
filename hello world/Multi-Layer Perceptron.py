#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)#載入數據
sess=tf.InteractiveSession()   #创建Interactive Session

in_units=784        #输入节点
h1_units=300        #隐含层输出节点
w1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))#初始化
b1=tf.Variable(tf.zeros([h1_units]))
w2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))

x=tf.placeholder(tf.float32,[None,in_units])   #输入的placeholder
keep_prob=tf.placeholder(tf.float32)           #dropout比率的placeholder

hidden1=tf.nn.relu(tf.matmul(x,w1)+b1)         #先进行ReLu 再dropout 最后进入softmax
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
y=tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)

#定义损失函数和选择优化器
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),
                                            reduction_indices=[1]))
train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)#使用Adadrad优化器

#计算步骤
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x: batch_xs,y_: batch_ys,keep_prob: 0.75})

#准确率评测
correct_prediction=tf.equal(tf.arg_max(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x: mnist.test.images,y_: mnist.test.labels,
                     keep_prob: 1.0}))