#!/usr/bin/python
# -*- coding: utf-8 -*-  
  
import os
import sys
import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data   
from PIL import Image  



# 声明图片宽高  
rows = 28  
cols = 28  
  
#创建会话  
sess = tf.Session()  
  
def mnistToimage(images, labels, save_dir, images_to_extract=8000):
    shape = sess.run(tf.shape(images))
    # 获取图片总数  
    images_count = shape[0]  
    pixels_per_image = shape[1]
    # 获取标签总数  
    shape = sess.run(tf.shape(labels))  
    labels_count = shape[0] 
    #一共有多少分类
    labels_classification_count = shape[1]
    labels = sess.run(tf.argmax(mnist.train.labels, 1))  
    
    if (images_count == labels_count):
        print ("Total %s images，%s lables" % (images_count, labels_count))  
        print ("每张图片包含 %s 个像素" % (pixels_per_image))  
        print ("数据类型：%s" % (mnist.train.images.dtype))    
    # mnist图像数据的数值范围是[0,1]，需要扩展到[0,255]，以便于人眼观看  
        if images.dtype == "float32":  
            print ("准备将数据类型从[0,1]转为binary[0,255]...")  
            for i in range(images_to_extract):  
                for n in range(pixels_per_image):  
                    if mnist.train.images[i][n] != 0:  
                        mnist.train.images[i][n] = 255  
                # 由于数据集图片数量庞大，转换可能要花不少时间，有必要打印转换进度  
                if ((i+1)%50) == 0:  
                    print ("图像浮点数值扩展进度：已转换 %s 张，共需转换 %s 张" % (i+1, images_to_extract))  

        # 创建数字图片的保存目录  
        for i in range(labels_classification_count):  
            dir = "%s/%s/" % (save_dir,i)  
            if not os.path.exists(dir):  
                print ("目录 ""%s"" 不存在！自动创建该目录..." % dir)  
                os.makedirs(dir)  

        # 通过python图片处理库，生成图片  
        indices = [0 for x in range(labels_classification_count)]  
        for i in range(images_to_extract):  
            img = Image.new("L",(cols,rows))  
            for m in range(rows):  
                for n in range(cols):  
                    img.putpixel((n,m), int(mnist.train.images[i][n+m*cols]))  
            # 根据图片所代表的数字label生成对应的保存路径  
            digit = labels[i]  
            path = "%s/%s/%s.bmp" % (save_dir, labels[i], indices[digit])  
            indices[digit] += 1  
            img.save(path)  
            # 由于数据集图片数量庞大，保存过程可能要花不少时间，有必要打印保存进度  
            if ((i+1)%50) == 0:  
                print ("图片保存进度：已保存 %s 张，共需保存 %s 张" % (i+1, images_to_extract))  
      
    else:  
        print ("图片数量和标签数量不一致！")

if __name__ == "__main__":
    # 读入mnist数据  
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
    # 当前路径下的保存目录  
    train_save_dir = "./mnist_digits_images/training_image"  
    test_save_dir = "./mnist_digits_images/test_image"  
    print("Training to Test image.........")
    mnistToimage(mnist.train.images, mnist.train.labels, train_save_dir)
    print("Save to Test image.........")
    mnistToimage(mnist.test.images, mnist.test.labels, test_save_dir)
    