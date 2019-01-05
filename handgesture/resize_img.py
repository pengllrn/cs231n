# -*- coding: utf-8 -*-
import tensorflow as tf
import scipy
import os

def resize_img(directory,size=[256,256]):
    filenames=os.listdir(directory)
    with tf.Session() as sess:
        for filename in filenames:
            _path=os.path.join(directory,filename)  #包含文件名的完整路径
            im=tf.gfile.FastGFile(_path,'rb').read()  #使用tensorflow获取该文件的句柄

            #输入im是 JPEG编码的图像，为string类型的tensor
            #将JPEG编码的图像解码为uint8张量,channels为图片的通道，可以为0，1，3.
            #当channels=0时，使用JPEG编码图像中的通道数。
            #ratio允许在解码期间通过整数因子缩小图像,允许的值为：1,2,4和8.这比稍后缩小图像要快得多。
            img_data=tf.image.decode_jpeg(im,channels=0,ratio=1)

            #转化图片数据类型的格式
            #image_float = tf.image.convert_image_dtype(im_data, tf.float32)

            #重新变换图片的大小
            #img_data为3-D和4-D的张量，size为变换后的尺寸
            #method为插值方法，有4种，分别是：双线性插值（Bilinear=0）,最近邻居法（NEAREST_NEIGHBOR=1）
            #双三次插值法（BICUBIC=2），面积插值法（AREA=3）
            #还有一种直接使用某种方法的函数，但对输入数据要求必须是4-D的，适合批量变换
            # tf.image.resize_area()
            resized=tf.image.resize_images(img_data,size=size,method=3)

            # 用此方法将执行所有先前的操作，这些操作将生成生成此张量的操作所需的输入。
            #可以喂入feed_dict
            resized=resized.eval(session=sess)

            #保存图片文件，可视化
            resize_dir="resize_to_%dx%d"%(size[0],size[1])
            if not os.path.exists(resize_dir):
                os.makedirs(resize_dir)
            scipy.misc.imsave(os.path.join(resize_dir,filename),resized)

if __name__ == '__main__':
    resize_img("picture",size=[64,64])