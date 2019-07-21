import tensorflow as tf
import numpy as np 
a=tf.placeholder(dtype=tf.float32,shape=[None,10,10,1])
f_1=tf.Variable(dtype=tf.float32,initial_value=tf.truncated_normal(shape=[3,3,1,2],dtype=tf.float32,mean=0,stddev=0.1))
conv=tf.nn.conv2d(input=a,filter=f_1,strides=[1,2,2,1],padding="SAME")
conv_1=tf.nn.conv2d(input=a,filter=f_1,strides=[1,1,1,1],padding="SAME")

kernel=tf.Variable(dtype=tf.float32,initial_value=tf.truncated_normal(shape=[3,3,1,2],dtype=tf.float32,mean=0,stddev=0.1))
deconv=tf.nn.conv2d_transpose(value=conv,filter=kernel,output_shape=[3,a.shape[1].value,a.shape[2].value,a.shape[3].value],strides=[1,2,2,1])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    b=np.random.rand(3,10,10,1)
    dec=sess.run(deconv,feed_dict={a:b})
    z=1