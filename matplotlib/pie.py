# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
labels=[u'学习习惯',u'学习态度',u'教师教学因素',u'课程自身难度'] #必须加u，不然会提示ascii错误
sizes=[30,20,15,35]
colors=['goldenrod','lightskyblue','yellow','red']
explode=[0,0,0,0.05]

patches,l_text,p_text=plt.pie(sizes,labels=labels,colors=colors,labeldistance=1.1,explode=explode,
                              autopct='%3.1f%%',shadow=False,startangle=90,pctdistance=0.6)
for t in l_text:
    t.set_size(30)

for t in p_text:
    t.set_size(20)

plt.legend(fontsize="large")
plt.show()