# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据
mnist=input_data.read_data_sets("MNist_data",one_hot=True)

#每个批次的大小
batch_size=100
#计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size

#命名空间
with tf.name_scope('input'):
    #定义两个placeholder
    x=tf.placeholder(tf.float32,[None,784])#输入
    y=tf.placeholder(tf.float32,[None,10])
    keep_prob=tf.placeholder(tf.float32)#dropout函数的保留率

#创建多层神经网络
with tf.name_scope('Input_Layer'):
    #输入层
    with tf.name_scope('Weight1'):
        W1=tf.Variable(tf.truncated_normal([784,200],stddev=0.1),name='W1')
    with tf.name_scope('b1'):
        b1=tf.Variable(tf.zeros([200])+0.1,name='b1')
    with tf.name_scope('Relu'):
        h1=tf.nn.relu(tf.matmul(x,W1)+b1,name='relu')
    with tf.name_scope('Dropout'):
        h1_drop=tf.nn.dropout(h1,keep_prob,name='dropout')
with tf.name_scope("Second_Layer"):
    #第二层
    W2=tf.Variable(tf.truncated_normal([200,100],stddev=0.1))
    b2=tf.Variable(tf.zeros([100])+0.1)
    h2=tf.nn.relu(tf.matmul(h1_drop,W2)+b2)
    h2_drop=tf.nn.dropout(h2,keep_prob)
with tf.name_scope("Third_Layer"):
    #第三层
    W3=tf.Variable(tf.truncated_normal([100,50],stddev=0.1))
    b3=tf.Variable(tf.zeros([50])+0.1)
    h3=tf.nn.relu(tf.matmul(h2_drop,W3)+b3)
    h3_drop=tf.nn.dropout(h3,keep_prob)
with tf.name_scope("Output_Layer"):
    #输出层
    W4=tf.Variable(tf.truncated_normal([50,10],stddev=0.1))
    b4=tf.Variable(tf.zeros([10])+0.1)
    prediction=tf.nn.softmax(tf.matmul(h3_drop,W4)+b4)

with tf.name_scope('Loss'):
    #损失函数
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

with tf.name_scope('Train'):
    #使用梯度下降
    train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

with tf.name_scope('Accuracy'):
    #结果放在一个布尔型列表中
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    #求准确率
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('logs/',sess.graph)
    for _ in range(11):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.8})
        #用测试集的数据进行测试，得到准确率
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("epoch "+str(_)+",Testting Accuracy: "+str(acc))