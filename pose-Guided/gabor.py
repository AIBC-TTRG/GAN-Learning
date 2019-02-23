# -*- coding: utf-8 -*-
    # gabor.py
    # 2015-7-7
    # github: https://github.com/michael92ht
    #__author__ = 'huangtao'
    
# import the necessary packages
import numpy as np
import cv2 as cv
from pylab import *

from trainer import *
from models256 import *
from datasets import deepfashion

#定义了一个4尺度6方向的Gabor变换
#并将4尺度6方向变换函数图像及指定图像变换后的图像保存在指定文件夹
#可扩展为各种方法求纹理特征值等
def Gabor_u4v6(image):
    #图像预处理
    data_format = 'NHWC'
    print('aaaaaaaaaaaaaaaaaa')
    print(image)
    # image = denorm_img(image, data_format)
    print('bbbbbbbbbbbbbbbb')
    print(image)
    # image=cv.imread(image,1)
    # sess=tf.Session() 
    # sess.run(tf.initialize_all_variables()) 
    # sess.run(tf.global_variables_initializer()) 
    # print("out1=",type(image)) 
    #转化为numpy数组 
    print('eeeeeeeeeeeeeeeeeeeeeeee')

    sess=tf.Session()
    print('fffffffffffffffffffffffff') 
    sess.run(tf.global_variables_initializer())
    print('gggggggggggggggggggggggg') 
    #转化为numpy数组 
    img_numpy=image.eval(session=sess) 
    # A = np(image)
    # print A
    print("out2=",type(img_numpy)) 
    # #转化为tensor 
    # img_tensor= tf.convert_to_tensor(img_numpy) 
    # print("out2=",type(img_tensor))


    # image=image.eval(session=sess) 
    # print("out2=",type(image))
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print('zzzzzzzzzzzzzzzzzzzzzzzzz')
    #     image = sess.run(image)
    print('ddddddddddddddddddddddddddddd')
    src = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print('cccccccccccccccccccccc')
    # print(src)
    src_f = np.array(src, dtype=np.float32)
    src_f /= 255.

    us=[7,12,17,21]             #4种尺度
    vs=[0,30,60,90,120,150]     #6个方向
    kernel_size =21             
    sig = 5                     #sigma 带宽，取常数5
    gm = 1.0                    #gamma 空间纵横比，一般取1
    ps = 0.0                    #psi 相位，一般取0
    i=0
    # dest_new=[]
    for u in us:
        for v in vs:
            print u
            print v
            lm = u
            th = v*np.pi/180
            kernel = cv.getGaborKernel((kernel_size,kernel_size),sig,th,lm,gm,ps)
            kernelimg = kernel/2.+0.5
            dest = cv.filter2D(src_f, cv.CV_32F,kernel)
            i+=1
            if i == 13:
                # cv.imwrite(image_save_path + str(i) + 'Kernel.jpg', cv.resize(kernelimg, (kernel_size*20,kernel_size*20))*256)
                # cv.imwrite(image_save_path + str(i) + 'Mag.jpg', np.power(dest,2))
                dest_numpy =  np.power(dest,2)
                img_tensor= tf.convert_to_tensor(dest_numpy)
                # img_tensor = tf.reshape(img_tensor, [128, 64]) 
                # dest_new = dest
            
            # cv.imwrite(image_save_path + str(i) + 'Kernel.jpg', cv.resize(kernelimg, (kernel_size*20,kernel_size*20))*256)
            # cv.imwrite(image_save_path + str(i) + 'Mag.jpg', np.power(dest,2))
                # i+=1
    return  img_tensor         
    
    # print dest_new
    # cv.imwrite(image_save_path + str(i) + 'Kernel.jpg', cv.resize(kernelimg, (kernel_size*20,kernel_size*20))*256)
    # cv.imwrite(image_save_path + str(i) + 'Mag.jpg', np.power(dest,2))


# if __name__ == '__main__':
#     image_save_path=r'/home/aibc/Desktop/gabor/111/'
#     image=r'/home/aibc/Desktop/gabor/1.png'
#     Gabor_u4v6(image,image_save_path)

