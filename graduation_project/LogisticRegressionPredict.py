# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import metrics
from compute_deprives import PreSampling,ComputeDepriveVarious

#数据处理，源数据情况
#文件名：data0_fine1_target.csv
#列名有： date,time,tmp_zmjc_1.zhenmianid,tmp_zmjc_1.zmtdid,
#tmp_zmjc_1.zmzjcid,tmp_zmjc_1.rowid,tmp_zmjc_1.colid,fd,xw,index,target,ajxw
#共12列，其中index为每个通道的索引，不区分不同的阵面
#时间区间范围是从3-22到4-1（10天），6-5到7-31（57天）。共67天，每天一组，<=4448条记录
row_p_data=pd.read_csv("E:/Pengliang/data/Achieve/data0/data0_fine1_target.csv")
data_train=row_p_data[row_p_data['date']<'2018-7-22'].copy()#7月22之前的用来训练，7月22之后的用作测试
data_test=row_p_data[row_p_data['date']>='2018-07-17'].copy()#前5天的时间会被抛弃
#开始计算衍生变量
psp_train=PreSampling(data_train)
psp_test=PreSampling(data_test)
cdv_train=ComputeDepriveVarious(psp_train,length=1)#预测1天的
cdv_test=ComputeDepriveVarious(psp_test,length=1)
ysbl_train=cdv_train.compute()
ysbl_test=cdv_test.compute()
print "compute finished!"

#建模第一步，数据维度的筛选
#1. 删除主键，索引，时间等与计算无关的维度
#2. 通过计算变量之间的相关系数，删去自变量与自变量之间相关性过高的变量
tmp_data=ysbl_train.drop(['date','index'],axis=1)
cor=np.abs(tmp_data.corr()) #这里遇到了一个bug，要把ysbl—先保存为csv文件，再取出才能正确的计算相关系数矩阵，详细过程已经删除。
#np.array(cor>0,dtype='i2')
#删去自变量之间相关系数绝对值大于0.85的
drop_list=['date','index','fd_3_min','fdc_3_avg','xwc_3_avg',
           'fdc_3_min','fd_5_min','fdc_5_avg','xwc_5_avg','fdc_5_min']
train_data=ysbl_train.drop(drop_list,axis=1)
test_data=ysbl_test.drop(drop_list,axis=1)

#手动分数据，也可以使用sklearn.model_selection帮助我们快速的划分，但是它会把数据弄乱，为了分析方便，先手工分
X_train_data=train_data.iloc[:,1:]
y_train_data=train_data.iloc[:,0]
X_test_data=test_data.iloc[:,1:]
y_test_data=test_data.iloc[:,0]
#标准化  Z-score标准化
ss=StandardScaler()
X_train_data=ss.fit_transform(X_train_data)
y_train_data=np.array(y_train_data,dtype='i2')
X_test_data=ss.fit_transform(X_test_data)
y_test_data=np.array(y_test_data,dtype='i2')

#模型训练
lgr=LogisticRegression(class_weight={1:9,0:1},n_jobs=-1)
lgr.fit(X_train_data,y_train_data)
#混淆矩阵
print metrics.confusion_matrix(y_test_data,lgr.predict(X_test_data))

from sklearn.metrics import  roc_curve,auc
import matplotlib.pyplot as plt
false_positive_rate,recall,thresholds=roc_curve(y_test_data,lgr.predict_proba(X_test_data)[:,1])
roc_auc=auc(false_positive_rate,recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,recall,'b',label='AUC = %0.2f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()

#超参数搜索
from sklearn.grid_search import GridSearchCV
parameters = {
    'C':{0.001,0.01,0.1,1,2},
    'tol':{0.001,0.01,0.05}
}
grid_search=GridSearchCV(lgr,parameters,n_jobs=-1,verbose=0,scoring='f1',cv=3)
grid_search.fit(X_train_data,y_train_data)
best_parameters=grid_search.best_estimator_.get_params()
print grid_search.best_params_
blgr=grid_search.best_estimator_
print metrics.confusion_matrix(y_test_data,blgr.predict(X_test_data))
