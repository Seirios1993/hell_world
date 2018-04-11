#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)#載入數據

sess=tf.InteractiveSession()

def weight_variable(shape):     #随机噪声 正态分布噪声
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):       #给偏置增加正值避免死亡节点
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):                #卷积层
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#第一个参数value：需要池化的输入，一般池化层接在卷积层后面，
# 所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape

#第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
# 因为我们不想在batch和channels上做池化，所以这两个维度设为了1

#第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]

#第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'

#返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
def max_pool_2x2(x):            #池化层
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x=tf.placeholder(tf.float32,[None,784])      #特征图像
y_=tf.placeholder(tf.float32,[None,10])      #真实的label
x_image=tf.reshape(x,[-1,28,28,1])           #由1D输入向量转化为2D图片结构

####第一个卷积层###
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)


####第二个卷积层###
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

####全连接层###
w_fc1=weight_variable([7*7*64,1024])        #7×7×64行 1024列
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])#变回1D向量  7×7×64为1张图片的全部特征，-1表示程序自己计算
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

####drop outv层###
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

####soft max层###
w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

####损失函数####
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),
                                            reduction_indices=[1]))
trian_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction=tf.equal(tf.arg_max(y_conv,1),tf.arg_max(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

####训练阶段####
tf.global_variables_initializer().run()
for i in range(20000):
    batch=mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],
                                                keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    trian_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))

