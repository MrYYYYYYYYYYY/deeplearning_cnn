import tensorflow as tf
import cv2
import os
import numpy as np
def Standerlize(picture):
        result=[]
        len_=len(picture)
        mean=0
        div=0
        for i in range(len_):
                mean=mean+picture[i]
        mean=mean/len_
        for i in range(len_):
                sub_div=(picture[i]-mean)*(picture[i]-mean)
                div=div+sub_div
        div=np.sqrt(div/len_)
        for i in range(len_):
                result.append((picture[i]-mean)/div)
        return result,mean,div 
with tf.device("/cpu:0"):
        list=os.listdir('./picture')
        picture=[50, 25, 76, 38, 19, 58, 29, 88, 44, 22, 11, 34, 17, 52, 26, 13, 40, 20]
        picture,mean,div=Standerlize(picture)
        picture=np.reshape(picture,[1,18])
        data=tf.placeholder(dtype=np.float32,shape=[None,18],name='input_data')
        W_1=tf.Variable(tf.random_normal(shape=[18,9],mean=0,stddev=0.1))
        b_1=tf.Variable(tf.random_normal(shape=[1,9],mean=0,stddev=0.1),dtype=np.float32)
        code=tf.Variable(tf.random_normal(shape=[9,2],mean=0,stddev=0.1),dtype=np.float32)
        b_2=tf.Variable(tf.random_normal(shape=[1,2],mean=0,stddev=0.1),dtype=np.float32)
        W_1_1=tf.Variable(tf.random_normal(shape=[9,18],mean=0,stddev=0.1))
        code_1=tf.Variable(tf.random_normal(shape=[2,9],mean=0,stddev=0.1),dtype=np.float32)
        b_3=tf.Variable(tf.random_normal(shape=[1,9],mean=0,stddev=0.1),dtype=np.float32)
        b_4=tf.Variable(tf.random_normal(shape=[1,18],mean=0,stddev=0.1),dtype=np.float32)
        layer_1=tf.add(tf.matmul(data,W_1),b_1)
        act_1=tf.nn.relu(layer_1)
        layer_2=tf.add(tf.matmul(act_1,code),b_2)
        act_2=tf.nn.relu(layer_2)

        re_layer_2=tf.add(tf.matmul(act_2,code_1),b_3)
        re_act_2=tf.nn.relu(re_layer_2)

        re_layer_1=tf.add(tf.matmul(re_act_2,W_1_1),b_4)

if __name__=="__main__":
        with tf.Session() as sess:
                rate=0.0001
                epoch=100
                out=re_layer_1
                c=out.shape.value
                loss=tf.reduce_mean(tf.multiply(tf.subtract(out,picture),tf.subtract(out,picture)))
                train_step=tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
                sess.run(tf.global_variables_initializer())
                for i in range(epoch):
                        total_loss=0
                        act_to=np.zeros(shape=[1,2],dtype=np.float32)
                        for j in range(100):
                                [_,loss_1,act]=sess.run([train_step,loss,act_2],feed_dict={data:picture})
                                total_loss+=loss_1
                                act_to=act_to+act
                        total_loss=total_loss/100
                        act_to=act_to/100
                        print('setp {},the loss: {},act:{}'.format(i, total_loss,act_to))
        

