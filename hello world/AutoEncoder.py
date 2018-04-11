#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(fan_in,fan_out,constant=1):            #xavier初始化器
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),
                             minval=low,maxval=high,
                             dtype=tf.float32)

#########################去噪自编码类#########################
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,
                optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input                           #输入变量数
        self.n_hidden = n_hidden                       #隐含层节点数
        self.transfer=transfer_function                #隐含层激活函数
        self.scale=tf.placeholder(tf.float32)
        self.training_scale=scale                      #高斯噪声系数
        network_weights=self._initialize_weights()     #参数初始化
        self.weights=network_weights

        #定义网络结构
        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden=self.transfer(tf.add(tf.matmul(    #transfer:激活函数
                          self.x+scale*tf.random_normal((n_input,)),
                          self.weights['w1']),self.weights['b1']))
        self.reconstruction=tf.add(tf.matmul(self.hidden,
                               self.weights['w2']),self.weights['b2'])
        #损失函数
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(  #损失函数形式
                          self.reconstruction,self.x),2.0))
        self.optimizer=optimizer.minimize(self.cost)     #优化器
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)                              #这句话在类实例建立时就完成了参数初始化

    #参数初始化函数
    def _initialize_weights(self):        # w1根据xavier初始化，其他置0
        all_weights=dict()                # 使用字典存放权值参数
        all_weights['w1']=tf.Variable(xavier_init(self.n_input,
                                                  self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                 dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,#保持输出与输入矩阵维度相同
                                self.n_input],dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                 dtype=tf.float32))
        return all_weights

    #一步训练 损失计算
    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),#sess.run()的好处，可以一步执行两个graph
            feed_dict={self.x: X,self.scale:self.training_scale})
        return cost
    #只求损失函数 用于测试集对模型性能进行评测
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x: X,
            self.scale: self.training_scale
        })
    #返回自编码器隐含层的特征，即一个获取抽象后特征的接口
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x: X,
            self.scale: self.training_scale
            })
    #将隐含层的输出作为输入，将高级特征复原为原始数据
    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.n_hidden: hidden})
    #整体运行一遍复原过程，包括transform和generate,输入原数据，输出复原数据
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x: X,
            self.scale: self.training_scale
            })
    #获取隐含层权重 w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    # 获取隐含层偏置系数 b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
#########################去噪自编码类#########################
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)  #載入數據
#数据标准化处理函数
def standard_scale(X_train,X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test
#随机block获取函数
def get_random_block_from_data(data,batch_size):
    start_index=np.random.randint(0,len(data)-batch_size)    #生成随机数
    return data[start_index:(start_index+batch_size)]
#对训练集 测试集进行标准化变换
X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)

#定义常用参数
n_samples=int(mnist.train.num_examples)
training_epochs=20
batch_size=128
display_step=1

#########################建立实例#########################
autoencoder=AdditiveGaussianNoiseAutoencoder(n_input=784,  #输入节点
                n_hidden=200,                              #隐含层节点数
                transfer_function=tf.nn.softplus,          #隐含层激活函数
                optimizer=tf.train.AdamOptimizer(learning_rate=0.001),#优化器学习速率
                scale=0.01)                                #噪声系数0.01

#########################训练过程#########################
for epoch in range(training_epochs):                       #20个epochs
    avg_cost=0.
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):                           #每个epochs里将所有训练数据迭代一次
        batch_xs=get_random_block_from_data(X_train,batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost+=cost/n_samples*batch_size

    if epoch%display_step==0:
        print("Epoch:",'%04d'%(epoch+1),"cost=",
              "{:.9f}".format(avg_cost))
print("Total cost: "+str(autoencoder.calc_total_cost((X_test))))