#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from trainer import *
from trainer import *
from trainer256 import *
from config import get_config
from utils import prepare_dirs_and_logger, save_config

# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# #按照第0维连接
# a = tf.concat( [t1, t2],0) 
# #按照第1维连接
# b = tf.concat([t1, t2],1) 

# print(t1)
# print(t2)
# print(a)
# print(b)

# c = b + b

# print(c)

# x = tf.constant([1, 4])
# y = tf.constant([2, 5])
# z = tf.constant([3, 6])
# xx = tf.stack([x,y,z]) 
# yy = tf.stack([x,y,z],axis=0) 
# zz = tf.stack([x,y,z],axis=1) 

# print(x)
# print(y)
# print(z)
# print(xx)
# print(yy)
# print(zz)

# sess=tf.Session()
# print('fffffffffffffffffffffffff') 
# sess.run(tf.glob1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# #按照第0维连接
# a = tf.concat( [t1, t2],0) 
# #按照第1维连接
# b = tf.concat([t1, t2],1) 

# print(t1)
# print(t2)
# print(a)
# print(b)

# c = b + b

# print(c)al_variables_initializer())
# print('gggggggggggggggggggggggg') 
# #转化为numpy数组 
# print b
# img_numpy=b.eval(session=sess) 
# print("out2=",type(img_numpy))
# print('aaaaaaaaaaaaaaa')
# print  img_numpy
# # #转化为tensor 
# img_tensor= tf.convert_to_tensor(img_numpy) 
# print("out2=",type(img_tensor))
# print('bbbbbbbbbbbbbbbbb')
# print (img_tensor) 

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print('zzzzzzzzzzzzzzzzzzzzzzzzz')
#     print (sess.run(c))

q = [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]]
w = tf.constant(q)
p = tf.strided_slice(q,[0,0,0],[1,1,1]) 
print(q) 
print(w)
 
print(w[0]) 
print(w[1])
print(w[2])  
print(p)
e = w[0]
r = w[1]
t = w[2]

print('***************')
print(tf.size(q))
print tf.size(w)
print(tf.size(e))
print tf.size(p)

print(type(w))
print(w.get_shape() )

ss = tf.constant('Hello, Tensor Flow!')
print(ss)

# t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
# a = tf.size(t)  # 12
# print(a)
# print(t)

with tf.Session() as sess:  
    print('wwwwwwwww')
    print(sess.run(w))

with tf.Session() as sess:  
    print('eeeeeeeeeee')
    print(sess.run(e)) 

with tf.Session() as sess:  
    print('rrrrrrrrrrrrrrr')
    print(sess.run(r)) 

with tf.Session() as sess:  
    print('tttttttttttttttt')
    print(sess.run(t))   