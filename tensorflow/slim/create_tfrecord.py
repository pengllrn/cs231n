# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import random
import math
import sys

_NUM_TEST=200  #验证集数量
_RANDOM_SEED=0  #随机种子
_NUM_SHARDS=5  #数据块
DATASET_DIR="E:/Pycharm/Project/NNetword/Tensorflow/slim/images"  #数据集路径
LABELS_FILENAME="E:/Pycharm/Project/NNetword/Tensorflow/slim/images/labels.txt"#标签路径

def _get_dataset_filename(datatset_dir,split_name,shard_id):
    """
    define the tfrecord file's path and name
    :param datatset_dir:
    :param split_name:
    :param shard_id:
    :return:
    """
    output_filename='image_%s_%04d-of-%04d.tfreocrd'%(split_name,shard_id,_NUM_SHARDS)
    return os.path.join(datatset_dir,output_filename)

def _data_exists(dataset_dir):
    """

    :param dataset_dir:
    :return:
    """
    for split_name in ['train','test']:
        for shard_id in range(_NUM_SHARDS):  #shard 碎片
            output_filename=_get_dataset_filename(dataset_dir,split_name,shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True

def _get_filename_and_classes(dataset_dir):
    directories=[]#数据目录
    class_names=[] #分类名称
    for filename in os.listdir(dataset_dir):
        path=os.path.join(dataset_dir,filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames=[]
    for directory in directories:
        for filename in  os.listdir(directory):
            path=os.path.join(directory,filename)
            photo_filenames.append(path)

    return photo_filenames,class_names

def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values=[values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data,image_format,class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded':bytes_feature(image_data),
        'image/format':bytes_feature(image_format),
        'image/class/label':int64_feature(class_id),
    }))

def write_label_file(labels_to_class_names,dataset_dir,filename=LABELS_FILENAME):
    labels_filename=os.path.join(dataset_dir,filename)
    with tf.gfile.Open(labels_filename,'w') as f:
        for label in labels_to_class_names:
            class_name=labels_to_class_names[label]
            f.write('%d:%s\n'%(label,class_name))

def _convert_dataset(split_name,filenames,class_names_to_ids,dataset_dir):
    assert split_name in ['train','test']
    num_per_shard=int(len(filenames)/_NUM_SHARDS)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename=_get_dataset_filename(dataset_dir,split_name,shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx=shard_id * num_per_shard
                    end_ndx=min((shard_id + 1)* num_per_shard,len(filenames)) #考虑不整除的情况
                    for i in range(start_ndx,end_ndx):
                        try:
                            #控制台输出
                            sys.stdout.write('\r>> Converting image %d/%d shard %d'% (i+1,len(filenames),shard_id))
                            sys.stdout.flush()
                            image_data=tf.gfile.FastGFile(filenames[i],'rb').read()
                            class_name=os.path.basename(os.path.dirname(filenames[i]))
                            class_id=class_names_to_ids[class_name]
                            exmple=image_to_tfexample(image_data,b'jpg',class_id)
                            tfrecord_writer.write(exmple.SerializeToString())
                        except IOError as e:
                            print("Could not read: ",filenames[i])
                            print("Error: ",e)
                            print("Skip it\n")

    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    if _data_exists(DATASET_DIR):
        print('tfrecord文件已经存在')
    else:
        photo_filenames,class_names=_get_filename_and_classes(DATASET_DIR)
        class_names_to_ids=dict(zip(class_names,range(len(class_names))))#[('daisy',0),('dandelion',1),...]
        #字典化 {'daisy':0,'dandelion':1,....}

        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)  #将列表随机排序
        training_filenames=photo_filenames[_NUM_TEST:] #分割用于测试和训练的图片
        testing_filenames=photo_filenames[:_NUM_TEST]

        #数据转化
        _convert_dataset('train',training_filenames,class_names_to_ids,DATASET_DIR)
        _convert_dataset('test',testing_filenames,class_names_to_ids,DATASET_DIR)

        label_to_class_names=dict(zip(range(len(class_names)),class_names))
        write_label_file(label_to_class_names,DATASET_DIR)


