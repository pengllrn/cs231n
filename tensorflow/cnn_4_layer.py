# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# sess = tf.InteractiveSession()

#数据集对象
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
n_batch=mnist.train.num_examples//100

#输入，参数
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
keep_prob=tf.placeholder("float")

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第一层卷积：输入层
#定义变量
x_image=tf.reshape(x,[-1,28,28,1])#conv2d函数的要求shape
W_conv1=weight_variable([5,5,1,32])#卷积核的大小
b_conv1=bias_variable([32])
#卷积-激活-池化
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#进行卷积，并通过激活函数
h_pool1=max_pool_2x2(h_conv1)#再进行池化

#第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64]) #32为上一次卷积核的输出厚度
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #N,7,7,64

#密集连接层：全连接层FC1.  全连接层的权重矩阵为2-D的
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#dropout
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#输出层：FC2
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
#把输出结果转化成概率
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

####模型评估
cross_entry=-tf.reduce_sum(y_*tf.log(y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entry)
correct_prediction=tf.equal(tf.argmax(y_,axis=1),tf.argmax(y_conv,axis=1))
accuracy_rate=tf.reduce_mean(tf.cast(correct_prediction,'float'))

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        train_step.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
        if i%100 == 0:
            print("now test accuracy is %g"%accuracy_rate.eval(feed_dict={
                x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0
            }))
