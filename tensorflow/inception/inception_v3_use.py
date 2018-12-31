# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

class NodeLookup(object):
    def __init__(self):
        label_lookup_path='inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_lookup='inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup=self.load(label_lookup_path,uid_lookup_lookup)

    def load(self,label_lookup_path,uid_lookup_path):
        proto_as_ascii_lines=tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human={}
        for line in proto_as_ascii_lines:
            line=line.strip('\n')#去掉换行符
            parsed_items=line.split('\t')
            uid=parsed_items[0]
            human_srting=parsed_items[1]
            uid_to_human[uid]=human_srting

        proto_as_ascii=tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid={}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class=int(line.split(': ')[1])  #获取字符串id
            if line.startswith('  target_class_string:'):
                target_class_string=line.split(': ')[1] #获取字符串
                node_id_to_uid[target_class]=target_class_string[1:-2]

        node_id_to_name={}
        for key,val in node_id_to_uid.items():
            name=uid_to_human[val]
            node_id_to_name[key]=name
        return node_id_to_name

    def id_to_string(self,node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

#创建一个图来存放google训练好的模型
with tf.gfile.GFile('inception_model/classify_image_graph_def.pb','rb') as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')

with tf.Session() as sess:
    # 'softmax:0'这个名字，可以在网络中找到这个节点，它的名字就是'(softmax)',
    softmax_tensor=sess.graph.get_tensor_by_name('softmax:0')#   ???
    for root,dirs,files in os.walk('images/'):
        for file in files:
            image_data=tf.gfile.FastGFile(os.path.join(root,file),'rb').read()
            prediction=sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})#图片需要是Jpg的
            # 运行softmax节点，向其中feed值
            # 可以在网络中找到这个名字，DecodeJpeg/contents，
            # 据此可以发现，根据名字取网络中op时，如果其名字带括号，就用括号内的名字，如果不带括号，就用右上角介绍的名字。
            # 而带个0，是默认情况，如果网络中出现同名节点，这个编号会递增
            predictions=np.squeeze(prediction) #把结果转换成1维数组

            image_path=os.path.join(root,file)
            print(image_path)

            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            #排序
            top_k=predictions.argsort()[-5:][::-1]#概率排序由小到大，再对这5个值作倒序
            node_lookup=NodeLookup()
            for node_id in top_k:
                human_string=node_lookup.id_to_string(node_id)
                score=predictions[node_id]
                print('%s (score = %.5f)'%(human_string,score))
            print()


