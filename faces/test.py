import tensorflow as tf
import cv2
import os
import detect_face
import numpy as np
import matplotlib.pyplot as plt
size=56
re_1=512
re=256
list=os.listdir('./picture')
picture=[]
list_0=os.listdir('./picture/faces_0')
for j in list_0:
        img=cv2.resize(src=cv2.imread('./picture/faces_0'+'/'+j,0),dsize=(size,size))
        img_dest=np.zeros(np.shape(img),dtype=np.float32)
        cv2.normalize(src=img, dst=img_dest, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_dest=np.reshape(img_dest,[1,size*size])
        picture.append(img_dest)

def Get_Face(cam_index):
    cap = cv2.VideoCapture(cam_index)
    num=0
    #建立模型
    with tf.Session() as sess:
        #建立模型框架   
        sess.run(tf.global_variables_initializer()) 
        saver = tf.train.import_meta_graph('ckpt/a.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('ckpt/'))
        graph = tf.get_default_graph()
        #tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
        x_=graph.get_tensor_by_name('input_data:0')
        code_=graph.get_tensor_by_name('code:0')
        cnn_output=graph.get_tensor_by_name('out:0')
        #获取Haar分类器
        classfier=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        #成功开启摄像头后捕获视频
        while   cap.isOpened() :
                #获取当前帧
                sucess,frame = cap.read()
                if sucess :
                        ##当前帧预处理
                        #灰度化
                        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                        ##检测人脸
                        faces=classfier.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3,minSize=(100,100))
                        #如果检测到人脸则截取并输出
                        if len(faces)>0 :
                                for face in faces:
                                        x,y,w,h = face
                                        #标识出人脸区域
                                        f=gray[y:y+h,x:x+w]  
                                        img=cv2.resize(f,dsize=(size,size))
                                        img_dest=np.zeros(np.shape(img),dtype=np.float32)
                                        cv2.normalize(src=img, dst=img_dest, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)       
                                        img_dest=np.reshape(img_dest,[1,size*size])
                                        f_out=sess.run(cnn_output,feed_dict={x_:img_dest})
                                        res=cv2.resize(np.reshape(f_out, (size, size))*255,dsize=(w,h))
                                        gray[y:y+h,x:x+w]=res

                #输出图像
                cv2.imshow('frame',gray)
                cv2.waitKey(1)
                #如果按下ESC则退出循环
                if cv2.waitKey(1) == 27:
                        break
        
    #关闭采集窗口，释放摄像头资源
    cv2.destroyAllWindows()
    cap.release()

def __minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield np.array(inputs)[excerpt]
def train(picture):
        with tf.device('/gpu:0'):
                data=tf.placeholder(dtype=np.float32,shape=[None,size*size],name='input_data')
                W_1=tf.Variable(tf.random_normal(shape=[size*size,re_1],mean=0,stddev=0.1))
                b_1=tf.Variable(tf.random_normal(shape=[1,re_1],mean=0,stddev=0.1),dtype=np.float32)
                code=tf.Variable(tf.random_normal(shape=[re_1,re],mean=0,stddev=0.1),dtype=np.float32)
                b_2=tf.Variable(tf.random_normal(shape=[1,re],mean=0,stddev=0.1),dtype=np.float32)
                b_3=tf.Variable(tf.random_normal(shape=[1,re_1],mean=0,stddev=0.1),dtype=np.float32)
                b_4=tf.Variable(tf.random_normal(shape=[1,size*size],mean=0,stddev=0.1),dtype=np.float32)
                W_1_1=tf.Variable(tf.random_normal(shape=[re_1,size*size],mean=0,stddev=0.1))
                b_1_1=tf.Variable(tf.random_normal(shape=[1,size*size],mean=0,stddev=0.1),dtype=np.float32)
                code_1=tf.Variable(tf.random_normal(shape=[re,re_1],mean=0,stddev=0.1),dtype=np.float32)
                b_2_1=tf.Variable(tf.random_normal(shape=[1,re_1],mean=0,stddev=0.1),dtype=np.float32)

                layer_1=tf.add(tf.matmul(data,W_1),b_1)

                act_1=tf.nn.relu(layer_1)

                layer_2=tf.add(tf.matmul(act_1,code),b_2)

                act_2=tf.nn.relu(layer_2,name='code')

                re_layer_2=tf.add(tf.matmul(act_2,code_1),b_2_1)

                re_act_2=tf.nn.relu(re_layer_2)

                re_layer_1=tf.add(tf.matmul(re_act_2,W_1_1),b_1_1,name='out')

        saver=tf.train.Saver(max_to_keep=1)

        with tf.Session() as sess:
                rate=0.0001
                epoch=100
                out=re_layer_1
                loss=tf.reduce_mean(tf.pow(data-out,2))
                train_step=tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
                sess.run(tf.global_variables_initializer())
                for i in range(epoch):
                        total_loss=0
                        act_to=np.zeros(shape=[1,re],dtype=np.float32)
                        for j in range(100):
                                [_,loss_1,act]=sess.run([train_step,loss,act_2],feed_dict={data:picture[j]})
                                total_loss+=loss_1
                                act_to+=act
                        total_loss=total_loss/100
                        act_to=act_to/100
                        print('setp {},the loss: {}'.format(i, total_loss))
                saver.save(sess,'ckpt/a.ckpt')
                _, a = plt.subplots(2, 10, figsize=(10, 2))

                for i in range(10):
                        a[0][i].imshow(np.reshape(picture[i], (size, size))*255)
                        a[1][i].imshow(np.reshape(sess.run(out,feed_dict={data:picture[i]}), (size,size))*255)
                plt.show()
if __name__=="__main__":
        train(picture)
        detect_face.detect_face()
        #Get_Face(0)


    
        

