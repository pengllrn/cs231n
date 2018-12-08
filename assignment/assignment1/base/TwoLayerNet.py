
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#初始化模型的参数向量，即得到一个初始模型
#input_size：W1的维度，为32*32*3
#hidden_size：隐层的长度
#output_size：输出的维度，即分类数
#b为偏移
def init_model(input_size, hidden_size, output_size, std=1e-4):
    model = {}
    model['W1'] = std * np.random.randn(input_size, hidden_size)
    model['b1'] = np.zeros(hidden_size)
    model['W2'] = std * np.random.randn(hidden_size, output_size)
    model['b2'] = np.zeros(output_size)
    return model
model=init_model(32*32*3,50,10)


# In[3]:


#定义一个两层的神经网络
def two_layer_net(X,y,model,reg):
    #两层神经网络的结构为：输入-FC-ReLu-FC-softmax-输出
    #模型参数
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    N, D = X.shape
    #前向传播
    h1=np.maximum(0,np.dot(X,W1) + b1)
    h2=np.dot(h1,W2) + b2
    scores=h2  #N*10
    #softmax and final loss
    #exp_class_scores为最终的输出值
    exp_class_scores=np.exp(scores)
    exp_correct_class_scores=exp_class_scores[np.arange(N),y]
    
    _loss=-np.log(exp_correct_class_scores/np.sum(exp_class_scores,axis=1))
    loss=sum(_loss)/N
    loss+=reg*(np.sum(W1**2)+np.sum(W2**2)) #L2正则项
    
    #gradient 
    grads={}
    #反向传播
    #误逆差传播（BP）
    #output-->softmax
    #h2对W2求偏

    dh2=exp_class_scores / np.sum(exp_class_scores,axis=1,keepdims=True)
    #dh2=scores / np.sum(scores,axis=1,keepdims=True)
    dh2[np.arange(N),y] -= 1
    dh2 /= N
    #FC
    #W2
    dW2=np.dot(h1.T,dh2)
    dW2 += 2*reg*W2

    db2=np.sum(dh2,axis=0)

    #layer1
    dh1=np.dot(dh2,W2.T)

    dW1X_b1 = dh1
    dW1X_b1[h1 <= 0] = 0

    dW1 = np.dot(X.T, dW1X_b1)
    dW1 += 2 * reg * W1

    db1 = np.sum(dW1X_b1, axis=0)
    
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['b1'] = db1
    
    return loss,grads


# In[4]:


#取数据
#先取20个数据来检查模型正确否
tr20=pd.read_csv("E:/Jupyter/data/tr20.csv")
na=np.array(tr20)
X_tr20=na[:,:3072]
y_tr20=na[:,3072]


# In[6]:


loss,grads=two_layer_net(X_tr20,y_tr20,model,0)
loss


# In[7]:


loss,grads=two_layer_net(X_tr20,y_tr20,model,500)
loss


# In[44]:


##正式开始训练
#X,y为训练集
#X_val,y_val为验证集
#model，模型初始量
#two_layer_net,两层神经网络的损失函数以及梯度模型
#learning_rate 学习率
#verbose：进度信息是否显示
#num_epochs:训练次数
#reg 正则强度
def predict(X,model):
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    
    h1 = np.maximum(0, np.dot(X, W1) + b1)
    h2 = np.dot(h1, W2) + b2
    scores = h2
    y_pred = np.argmax(scores, axis=1)
    return y_pred
def train(X, y, X_val, y_val,
    model,two_layer_net,
    num_epochs,reg,
    learning_rate=1e-3, learning_rate_decay=0.95,verbose=True):
    
    for it in range(num_epochs):
        loss, grads = two_layer_net(X,y,model,reg=reg)
        #更新模型
        for param_name in model:
            model[param_name] += -learning_rate * grads[param_name]

        #训练准确率
        train_acc = (predict(X,model) == y).mean()
        val_acc = (predict(X_val,model) == y_val).mean()
        #显示训练进度
        if verbose and it % 10 == 0:
            print('Finished epoch %d / %d: loss %f, train_acc: %f, val_acc: %f' % (it, num_epochs, loss,train_acc,val_acc))


# In[9]:


#取50个数据作为验证集
val50=pd.read_csv("E:/Jupyter/data/val50.csv")
na=np.array(val50)
X_val50=na[:,:3072]
y_val50=na[:,3072]
#取1000个用于训练的数据
tr1000=pd.read_csv("E:/Jupyter/data/tr1000.csv")
na=np.array(tr1000)
X_tr1000=na[:,:3072]
y_tr1000=na[:,3072]


# In[54]:


#先检验一下，模型是不是正确的
reg=0
lr=1e-4
model=init_model(32*32*3,50,10)
train(X_tr20,y_tr20,X_val50,y_val50,model,two_layer_net,300,reg,lr)


# In[ ]:


#很好，产生过拟合，训练集的精度为100%，说明模型没问题


# In[46]:


#正式训练
#第一步：调一个合适的学习率
reg=0
lr=2*1e-4
model=init_model(32*32*3,50,10)
train(X_tr1000,y_tr1000,X_val50,y_val50,model,two_layer_net,300,reg,lr)


# In[52]:


reg=500
lr=3*1e-4
model=init_model(32*32*3,50,10)
train(X_tr1000,y_tr1000,X_val50,y_val50,model,two_layer_net,300,reg,lr)


# In[48]:


print(predict(X_tr20,model))


# In[49]:


print(y_tr20)

