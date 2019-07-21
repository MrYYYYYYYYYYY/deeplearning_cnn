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
        #######     parameter
        TYPE_NUM=21
        BATCH_SIZE=50
        INPUT_SIZE=192
        CHANNEL_1=32
        CHANNEL_2=64
        CHANNEL_3=128
        CHANNEL_4=256
        CHANNEL_5=512
        CHANNEL_6=4096
        CHANNEL_7=4096
        CHANNEL_8=21
        CHANNEL_D_1=21
        CHANNEL_D_2=21
        IMG_W=224
        IMG_H=224
        #######     model
        #input
        input_1=tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,INPUT_SIZE,INPUT_SIZE,3],name="input")
        ## encoder
        #conv_1+pool_1
        filter_1=tf.Variable(initial_value=tf.truncated_normal(shape=[3,3,3,CHANNEL_1],dtype=tf.float32,mean=0,stddev=0.1),name="filter_1")
        bias_1=tf.Variable(initial_value=tf.truncated_normal(shape=[CHANNEL_1],dtype=tf.float32,mean=0,stddev=0.1),name="bias_1")
        conv_1=tf.nn.bias_add(value=tf.nn.conv2d(input=input_1,filter=filter_1,strides=[1,1,1,1],padding="SAME"),bias=bias_1,name="conv_1")
        active_1=tf.nn.relu(features=conv_1,name="active_1")
        result_1=tf.nn.max_pool(value=active_1,padding="SAME",ksize=[1,2,2,1],strides=[1,2,2,1],name="result_1")
        #conv_2+pool_2
        filter_2=tf.Variable(initial_value=tf.truncated_normal(shape=[3,3,CHANNEL_1,CHANNEL_2],dtype=tf.float32,mean=0,stddev=0.1),name="filter_2")
        bias_2=tf.Variable(initial_value=tf.truncated_normal(shape=[CHANNEL_2],dtype=tf.float32,mean=0,stddev=0.1),name="bias_2")
        conv_2=tf.nn.bias_add(value=tf.nn.conv2d(input=result_1,filter=filter_2,strides=[1,1,1,1],padding="SAME"),bias=bias_2,name="conv_2")
        active_2=tf.nn.relu(features=conv_2,name="active_2")
        result_2=tf.nn.max_pool(value=active_2,padding="SAME",ksize=[1,2,2,1],strides=[1,2,2,1],name="result_2")
        #conv_3+pool_3
        filter_3=tf.Variable(initial_value=tf.truncated_normal(shape=[3,3,CHANNEL_2,CHANNEL_3],dtype=tf.float32,mean=0,stddev=0.1),name="filter_3")
        bias_3=tf.Variable(initial_value=tf.truncated_normal(shape=[CHANNEL_3],dtype=tf.float32,mean=0,stddev=0.1),name="bias_3")
        conv_3=tf.nn.bias_add(value=tf.nn.conv2d(input=result_2,filter=filter_3,strides=[1,1,1,1],padding="SAME"),bias=bias_3,name="conv_3")
        active_3=tf.nn.relu(features=conv_3,name="active_3")
        result_3=tf.nn.max_pool(value=active_3,padding="SAME",ksize=[1,2,2,1],strides=[1,2,2,1],name="result_3")
        #conv_4+pool_4
        filter_4=tf.Variable(initial_value=tf.truncated_normal(shape=[3,3,CHANNEL_3,CHANNEL_4],dtype=tf.float32,mean=0,stddev=0.1),name="filter_4")
        bias_4=tf.Variable(initial_value=tf.truncated_normal(shape=[CHANNEL_4],dtype=tf.float32,mean=0,stddev=0.1),name="bias_4")
        conv_4=tf.nn.bias_add(value=tf.nn.conv2d(input=result_3,filter=filter_4,strides=[1,1,1,1],padding="SAME"),bias=bias_4,name="conv_4")
        active_4=tf.nn.relu(features=conv_4,name="active_4")
        result_4=tf.nn.max_pool(value=active_4,padding="SAME",ksize=[1,2,2,1],strides=[1,2,2,1],name="result_4")
        #conv_5+pool_5
        filter_5=tf.Variable(initial_value=tf.truncated_normal(shape=[3,3,CHANNEL_4,CHANNEL_5],dtype=tf.float32,mean=0,stddev=0.1),name="filter_5")
        bias_5=tf.Variable(initial_value=tf.truncated_normal(shape=[CHANNEL_5],dtype=tf.float32,mean=0,stddev=0.1),name="bias_5")
        conv_5=tf.nn.bias_add(value=tf.nn.conv2d(input=result_4,filter=filter_5,strides=[1,1,1,1],padding="SAME"),bias=bias_5,name="conv_5")
        active_5=tf.nn.relu(features=conv_5,name="active_5")
        result_5=tf.nn.max_pool(value=active_5,padding="SAME",ksize=[1,2,2,1],strides=[1,2,2,1],name="result_5")
        #conv_6 instead of FC layer
        '''
        shape_h=result_5.shape[1].value
        shape_w=result_5.shape[2].value
        filter_6=tf.Variable(initial_value=tf.truncated_normal(shape=[shape_h,shape_w,CHANNEL_6],dtype=tf.float32,mean=0,stddev=0.1),name="filter_6")
        '''
        filter_6=tf.Variable(initial_value=tf.truncated_normal(shape=[1,1,CHANNEL_5,CHANNEL_6],dtype=tf.float32,mean=0,stddev=0.1),name="filter_6")
        bias_6=tf.Variable(initial_value=tf.truncated_normal(shape=[CHANNEL_6],dtype=tf.float32,mean=0,stddev=0.1),name="bias_6")
        conv_6=tf.nn.bias_add(value=tf.nn.conv2d(input=result_5,filter=filter_6,strides=[1,1,1,1],padding="SAME"),bias=bias_6,name="conv_6")
        active_6=tf.nn.relu(features=conv_6,name="active_6")
        #conv_7
        filter_7=tf.Variable(initial_value=tf.truncated_normal(shape=[1,1,CHANNEL_6,CHANNEL_7],dtype=tf.float32,mean=0,stddev=0.1),name="filter_7")
        bias_7=tf.Variable(initial_value=tf.truncated_normal(shape=[CHANNEL_7],dtype=tf.float32,mean=0,stddev=0.1),name="bias_7")
        conv_7=tf.nn.bias_add(value=tf.nn.conv2d(input=active_6,filter=filter_7,strides=[1,1,1,1],padding="SAME"),bias=bias_7,name="conv_7")
        active_7=tf.nn.relu(features=conv_7,name="active_7")
        #conv_8
        filter_8=tf.Variable(initial_value=tf.truncated_normal(shape=[1,1,CHANNEL_7,CHANNEL_8],dtype=tf.float32,mean=0,stddev=0.1),name="filter_8")
        bias_8=tf.Variable(initial_value=tf.truncated_normal(shape=[CHANNEL_8],dtype=tf.float32,mean=0,stddev=0.1),name="bias_8")
        conv_8=tf.nn.bias_add(value=tf.nn.conv2d(input=active_7,filter=filter_8,strides=[1,1,1,1],padding="SAME"),bias=bias_8,name="conv_8")
        active_8=tf.nn.relu(features=conv_8,name="active_8")
        ###decoder
        #deconv_1
        filter_D_1=tf.Variable(initial_value=tf.truncated_normal(shape=[8,8,CHANNEL_D_1,CHANNEL_8],dtype=tf.float32,mean=0,stddev=0.1),name="filter_D_1")
        deconv_1=tf.nn.conv2d_transpose(value=active_8,filter=filter_D_1,strides=[1,4,4,1],output_shape=[active_8.shape[0].value,active_4.shape[1].value,active_4.shape[2].value,TYPE_NUM],padding="SAME")
        #deconv_2
        filter_D_2=tf.Variable(initial_value=tf.truncated_normal(shape=[16,16,CHANNEL_D_2,CHANNEL_D_1],dtype=tf.float32,mean=0,stddev=0.1),name="filter_D_1")
        deconv_2=tf.nn.conv2d_transpose(value=deconv_1,filter=filter_D_2,strides=[1,8,8,1],output_shape=[active_8.shape[0].value,IMG_H,IMG_W,TYPE_NUM],padding="SAME")

if __name__=="__main__":
        
        with tf.Session() as sess:
                
                rate=0.0001
                epoch=100
                out=deconv_2
                '''
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
                '''

        

