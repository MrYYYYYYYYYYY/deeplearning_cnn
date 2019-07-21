import tensorflow as tf
import cv2
import os 
import numpy as np
import skimage
import matplotlib.pyplot as plt
import time
import tqdm
#定义类别名称
__list=['yinxiang','others']
size=160
def img_show(img):
    cv2.imshow("img",img)
    if(cv2.waitKey(1)==27):
        cv2.destroyAllWindows()
#类别名称修改
def set_name(str_):
    __list[0]=str_
#样本读入并生成标注，打乱顺序
def __load_data(Dir,Rate):
    #样本绝对路径
    #读入样本并将其标准化
    dir_r=os.listdir(Dir)
    data_p=[]
    label_p=[]
    len_total=0
    for j,dir_f in enumerate(dir_r): 
        print('-----------------------file------------------- :%s'%(dir_f))
        dir_p=os.listdir(Dir+'/'+dir_f)
        len_dir=80
        #len_total=len_total+len_dir
        len_total=len_total+len_dir
        for i in range (len_dir):
            try:
                path_p=Dir+'/'+dir_f+'/'+dir_p[i]
            except BaseException:
                print('错误，当前图像操作失败')
            #读入样本并灰度化
            print('----------------reading image :%s'%(dir_p[i]))
            data=cv2.imread(path_p,0)
            m,n=np.shape(data)
            #图像统一为64*64*3并进行归一化
            #直方图均衡
            data=cv2.equalizeHist(data)
            data=cv2.resize(data,(size,size))
            result = np.zeros(data.shape, dtype=np.float32)
            #图像归一化
            cv2.normalize(data, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            result=np.reshape(result,[size,size,1])
            data_p.append(result)
            #生成标签
            label_p.append(int(j))
    #打乱数据顺序
    num=np.shape(data_p)[0]
    data_p=np.asarray(data_p,np.float32)
    label_p=np.asarray(label_p,np.int32)
    label=np.arange(num)
    np.random.shuffle(label)
    np.random.shuffle(label)
    np.random.shuffle(label)
    data_p=data_p[label]
    label_p=label_p[label]
    #按照Rate分配训练集与测试集
    train_data_p=data_p[:int(Rate*len_total)]
    train_label_p=label_p[:int(Rate*len_total)]
    test_data_p=data_p[int(Rate*len_total)+1:len_total]
    test_label_p=label_p[int(Rate*len_total)+1:len_total]
    return train_data_p,train_label_p,test_data_p,test_label_p
#Dir='C:/Users/yinxiang/AppData/Roaming/Code/User/faces_recognize/picture/faces_0'
#C:\Users\yinxiang\AppData\Roaming\Code\User\faces_recognize\picture
def show_img(picture):
    img=np.reshape(picture,[size,size])
    plt.imshow(img,cmap='gray')
    plt.show()
def get_data(Dir): 
    dir_r=os.listdir(Dir)
    data_p=[]
    label_p=[]
    len_dir=len(dir_r)
    for i in range (len_dir):
        path_p=Dir+'/'+dir_r[i]
        #读入样本
        #print('----------------reading image :%s'%(dir_r[i]))
        data=cv2.imread(path_p,0)
        #图像统一为64*64*3并进行归一化
        data=cv2.resize(data,(size,size))
        #直方图均衡化
        data=cv2.equalizeHist(data)
        result = np.zeros(data.shape, dtype=np.float32)
        cv2.normalize(data, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        result=np.reshape(result,[size,size,1])
        data_p.append(result)
        #生成标签
        label_p.append(4)
    return data_p
#定义数据分批函数
#数据增强
def data_enhancement(data,label):
    num,x,y,_=np.shape(data)
    center_x=int(x*0.5)
    center_y=int(y*0.5)
    enhanced_data=[]
    enhanced_label=[]
    for i in range(num):
        #加入原始数据
        enhanced_data.append(data[i])
        enhanced_label.append(label[i])
        #图像水平翻转
        enhanced_data.append(np.reshape(cv2.flip(data[i], 1),[x,y,1]))
        enhanced_label.append(label[i])
        #图像加高斯噪声
        enhanced_data.append(skimage.util.random_noise(data[i],mode='gaussian',seed=None,clip=True))
        enhanced_label.append(label[i])
        #图像仿射变换
        #enhanced_label.append(label[i])
        #数据随机剪裁
        size=96
        ran_x=np.random.randint(34)+30
        ran_y=np.random.randint(32)
        img=data[i][ran_x:ran_x+size,ran_y:ran_y+size]
        enhanced_data.append(np.reshape(cv2.resize(img,(x,y)),[x,y,1]))
        enhanced_label.append(label[i])
        #亮度调节-15%~15%
        img=np.zeros([x,y])
        img=data[i]+(np.random.rand()-0.5)*0.3
        img[img>1]=1
        img[img<0]=0
        enhanced_data.append(np.reshape(img,[x,y,1]))
        enhanced_label.append(label[i])
        #图像小角度旋转 -20°~20°
        rot_mat = cv2.getRotationMatrix2D((center_x,center_y),np.random.randint(20)-10, 1.0)
        enhanced_data.append(np.reshape(cv2.warpAffine(data[i],rot_mat,(x,y)),[x,y,1]))
        enhanced_label.append(label[i])
        for j in range(4):
            str_='img_'+str(i*4+j)+'.jpg'
            cv2.imwrite('C:/Users/yinxiang/AppData/Roaming/Code/User/faces_recognize/data/'+str_,enhanced_data[i*4+j]*255)
    #打乱数据顺序
    num=np.shape(enhanced_data)[0]
    enhanced_data=np.asarray(enhanced_data,np.float32)
    enhanced_label=np.asarray(enhanced_label,np.int32)
    label=np.arange(num)
    np.random.shuffle(label)
    enhanced_data=enhanced_data[label]
    enhanced_label=enhanced_label[label]
    return enhanced_data,enhanced_label
def __minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
#list->one_hot转换函数
def __one_hot(labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels],1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 6]), 1.0, 0.0)
    return onehot_labels 
#cnn模型nnum为分类数量.模型结构:Conv-MaxPool-Conv-MaxPool-dropout-Conv-MaxPool-Conv-MaxPool-dropout-flaten-dense-output
def train_cnn_model(num):
    with tf.name_scope('graph') as scope:
        #定义占位符
        __dropout=tf.placeholder(name='drop_out',dtype='bool',shape=[1])
        __train_label=tf.placeholder(name='train_label',dtype=tf.int32,shape=[None,])
        __train_data=tf.placeholder(name='train_data',dtype=tf.float32,shape=[None,size,size,1])
        ##加入dropout模型
        #卷积层_1
        layer_1=tf.layers.conv2d(name='conv_1',
                                inputs=__train_data,
                                filters=32,
                                kernel_size=[3,3],
                                activation=tf.nn.relu,
                                padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),)
        #池化层_1  128->64
        pooling_1=tf.layers.max_pooling2d(name='max_pooling_1',
                                        inputs=layer_1,
                                        pool_size=[2,2],
                                        strides=2)
        #卷积层_2
        layer_2=tf.layers.conv2d(name='conv_2',
                                inputs=pooling_1,
                                filters=32,
                                kernel_size=[3,3],
                                activation=tf.nn.relu,
                                padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        #池化层_2   64->32
        pooling_1=tf.layers.max_pooling2d(name='max_pooling_2',
                                        inputs=layer_2,
                                        pool_size=[2,2],
                                        strides=2)
        #dropout防止过拟合
        dropout_1=tf.layers.dropout(name='dropout_1',
                                    rate=0.1,
                                    inputs=pooling_1,
                                    training=__dropout[0])
        #卷积层_3
        layer_3=tf.layers.conv2d(name='conv_3',
                                inputs=dropout_1,
                                filters=64,
                                kernel_size=[3,3],
                                activation=tf.nn.relu,
                                padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        #池化层_3   32->16
        pooling_3=tf.layers.max_pooling2d(name='max_pooling_3',
                                        inputs=layer_3,
                                        pool_size=[2,2],
                                        strides=2)
        #卷积层_4
        layer_4=tf.layers.conv2d(name='conv_4',
                                inputs=pooling_3,
                                filters=64,
                                kernel_size=[3,3],
                                activation=tf.nn.relu,
                                padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        #池化层_4  16->8
        pooling_2=tf.layers.max_pooling2d(name='max_pooling_2',
                                        inputs=layer_4,
                                        pool_size=[2,2],
                                        strides=2)
        #dropout防止过拟合
        dropout_2=tf.layers.dropout(name='dropout_2',
                                    rate=0.2,
                                    inputs=pooling_2,
                                    training=__dropout[0])
        #flaten层
        reshape_1 = tf.layers.flatten(dropout_2)
        #全连接层_1
        dense_1=tf.layers.dense(name='fully_connected_1',
                                inputs=reshape_1,
                                units=1024,
                                activation=tf.nn.relu,
                                use_bias=True,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        #dropout防止过拟合
        dropout_3=tf.layers.dropout(name='dropout_3',
                                    rate=0.2,
                                    inputs=dense_1,
                                    training=__dropout[0])
        #全连接层_1
        dense_2=tf.layers.dense(name='fully_connected_2',
                                inputs=dropout_3,
                                units=512,
                                activation=tf.nn.relu,
                                use_bias=True,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        #dropout防止过拟合
        dropout_4=tf.layers.dropout(name='dropout_4',
                                    rate=0.5,
                                    inputs=dense_2,
                                    training=__dropout[0])
        #输出层
        output=tf.layers.dense(name='output',
                                inputs=dropout_4,
                                units=2,
                                activation=None,
                                use_bias=True,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.003))
    #定义训练次数
    Rate=0.0001
    epoch=100
    cnn_output=output
    #定义交叉熵损失函数
    cross_entropy=tf.losses.sparse_softmax_cross_entropy(labels=__train_label,logits=cnn_output)
    train_step=tf.train.AdamOptimizer(learning_rate=Rate).minimize(cross_entropy)
    Accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(cnn_output, 1), tf.int32),__train_label),tf.float32))
    #保存最近一次模型与精度
    saver=tf.train.Saver(max_to_keep=1)
    f=open('ckpt/Accuracy.txt','w')
    #读入训练样本
    train_data_p,train_label_p,test_data_p,test_label_p=__load_data(Dir='picture',len_dir=100,Rate=0.8)
    #会话GPU设置
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction=0.8 # 程序最多只能占用指定gpu50%的显存
    #config.gpu_options.allow_growth = True      #程序按需申请内存
    #with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        #保存计算图
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range (epoch):
            #平均准确率与最大准确率
            accuracy_total=0
            accuracy_max=0
            #平均损失
            loss_total=0
            #batch总数
            n_batch=0
            #batch大小为20,训练模型
            for batch_train_data,batch_train_label in __minibatches(inputs=train_data_p,targets=train_label_p,batch_size=20,shuffle=True):
                #图像增强
                batch_train_data_enhanced,batch_train_label_enhanced=data_enhancement(batch_train_data,batch_train_label)
                [_,train_accuracy,loss]=sess.run([train_step,Accuracy,cross_entropy],feed_dict = {__train_data:batch_train_data_enhanced,__train_label: batch_train_label_enhanced,__dropout:[True]})
                n_batch=n_batch+1
                loss_total=loss_total+loss
                accuracy_total=accuracy_total+train_accuracy
            if i % 1 == 0:
                accuracy_total=accuracy_total/n_batch
                loss_total=loss_total/n_batch
                test_accuracy = Accuracy.eval(feed_dict = {__train_data: test_data_p,__train_label:test_label_p,__dropout:[False]})
                #记录最高准确率,保存准确率最高的模型并记录准确率
                f.write(str(i+1)+', val_acc: '+str(test_accuracy)+'\n')
                if accuracy_max<=accuracy_total:
                    accuracy_max=accuracy_total
                print('setp {},the train accuracy: {}'.format(i, accuracy_total))
                print('--------the test accuracy :{}'.format(test_accuracy))
                pp_4='C:/Users/yinxiang/AppData/Roaming/Code/User/faces_recognize/faces_4'
                pp_5='C:/Users/yinxiang/AppData/Roaming/Code/User/faces_recognize/faces_5'
                pp_6='C:/Users/yinxiang/AppData/Roaming/Code/User/faces_recognize/faces_6'
                pp_7='C:/Users/yinxiang/AppData/Roaming/Code/User/faces_recognize/faces_7'
                pp_8='C:/Users/yinxiang/AppData/Roaming/Code/User/faces_recognize/faces_8'
                dd_4=get_data(pp_4)
                ddd_4=[]
                for i in range(len(os.listdir(pp_4))):
                    ddd_4.append(0)
                res_4=tf.argmax(tf.nn.softmax(sess.run(cnn_output,feed_dict={__train_data: dd_4,__dropout:[False]})),1)
                d_4=Accuracy.eval(feed_dict = {__train_data: dd_4,__train_label:ddd_4,__dropout:[False]})
                print('-------- accuracy_4 :{}'.format(d_4))
                f.write(str(i+1)+', val_acc———4: '+str(d_4)+'\n')
                dd_5=get_data(pp_5)
                ddd_5=[]
                for i in range(len(os.listdir(pp_5))):
                    ddd_5.append(10)
                res_5=tf.argmax(tf.nn.softmax(sess.run(cnn_output,feed_dict={__train_data: dd_5,__dropout:[False]})),1)
                d_5=Accuracy.eval(feed_dict = {__train_data: dd_5,__train_label:ddd_5,__dropout:[False]})
                print('-------- accuracy_5 :{}'.format(d_5))
                f.write(str(i+1)+', val_acc———5: '+str(d_5)+'\n')
                dd_6=get_data(pp_6)
                ddd_6=[]
                for i in range(len(os.listdir(pp_6))):
                    ddd_6.append(1)
                res_6=tf.argmax(tf.nn.softmax(sess.run(cnn_output,feed_dict={__train_data: dd_6,__dropout:[False]})),1)
                d_6=Accuracy.eval(feed_dict = {__train_data: dd_6,__train_label:ddd_6,__dropout:[False]})
                print('-------- accuracy_6 :{}'.format(d_6))
                f.write(str(i+1)+', val_acc———6: '+str(d_6)+'\n')
                dd_7=get_data(pp_7)
                ddd_7=[]
                for i in range(len(os.listdir(pp_7))):
                    ddd_7.append(0)
                res_7=tf.argmax(tf.nn.softmax(sess.run(cnn_output,feed_dict={__train_data: dd_7,__dropout:[False]})),1)
                d_7=Accuracy.eval(feed_dict = {__train_data: dd_7,__train_label:ddd_7,__dropout:[False]})
                f.write(str(i+1)+', val_acc———7: '+str(d_7)+'\n')
                print('-------- accuracy_7 :{}'.format(d_7))

                dd_8=get_data(pp_8)
                ddd_8=[]
                for i in range(len(os.listdir(pp_8))):
                    ddd_8.append(0)
                res_8=tf.argmax(tf.nn.softmax(sess.run(cnn_output,feed_dict={__train_data: dd_8,__dropout:[False]})),1)
                d_8=Accuracy.eval(feed_dict = {__train_data: dd_8,__train_label:ddd_8,__dropout:[False]})
                f.write(str(i+1)+', val_acc———7: '+str(d_8)+'\n')
                print('-------- accuracy_7 :{}'.format(d_8))
                if accuracy_max>=0.95:
                    saver.save(sess,'ckpt/CNN_faces_recognition.ckpt',global_step=1) 
                    break
            #准确率大于0.99则保存模型结束训练
    f.close()
    sess.close()
#GPU加速cnn模型
def trian_gpu_model():
    cpu='/cpu:0'
    gpu='/gpu:0'
    with tf.device(cpu):
        #定义占位符与变量
        __keepprob=tf.placeholder(name='drop_out',dtype=tf.float32,shape=[2])
        __train_label=tf.placeholder(name='train_label',dtype=tf.int32,shape=[None,])
        __train_data=tf.placeholder(name='train_data',dtype=tf.float32,shape=[None,size,size,1])
        kernel_1=tf.get_variable(name='kernel_1',
                                shape=[3,3,1,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_1=tf.get_variable(name='Bais_1',
                                shape=[1,1,1,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        kernel_2=tf.get_variable(name='kernel_2',
                                shape=[3,3,32,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_2=tf.get_variable(name='Bais_2',
                                shape=[1,1,1,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        kernel_3=tf.get_variable(name='kernel_3',
                                shape=[3,3,32,64],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_3=tf.get_variable(name='Bais_3',
                                shape=[1,1,1,64],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        kernel_4=tf.get_variable(name='kernel_4',
                                shape=[3,3,64,64],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_4=tf.get_variable(name='Bais_4',
                                shape=[1,1,1,64],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        kernel_5=tf.get_variable(name='kernel_5',
                                shape=[3,3,64,128],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_8=tf.get_variable(name='Bais_8',
                                shape=[1,1,1,128],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        w_1=tf.get_variable(name='w_1',
                            shape=[5*5*128,1024],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        w_2=tf.get_variable(name='w_2',
                            shape=[1024,512],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        w_3=tf.get_variable(name='w_3',
                            shape=[512,2],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        Bais_5=tf.get_variable(name='Bais_5',
                                shape=[1,1024],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_6=tf.get_variable(name='Bais_6',
                                shape=[1,512],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_7=tf.get_variable(name='Bais_7',
                                shape=[1,2],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
    with tf.device(gpu):
        #卷积
        Conv_1=tf.nn.conv2d(name='conv_1',
                            input=__train_data,
                            filter=kernel_1,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_1=tf.add(Conv_1,Bais_1)
        activation_1=tf.nn.relu(Layer_1)
        #池化
        pooling_1=tf.nn.pool(name='pooling_1',
                            input=activation_1,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #卷积
        Conv_2=tf.nn.conv2d(name='conv_2',
                            input=pooling_1,
                            filter=kernel_2,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_2=tf.add(Conv_2,Bais_2)
        activation_2=tf.nn.relu(Layer_2)
        #池化
        pooling_2=tf.nn.pool(name='pooling_2',
                            input=activation_2,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #卷积
        Conv_3=tf.nn.conv2d(name='conv_3',
                            input=pooling_2,
                            filter=kernel_3,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_3=tf.add(Conv_3,Bais_3)
        activation_3=tf.nn.relu(Layer_3)
        #池化
        pooling_3=tf.nn.pool(name='pooling_3',
                            input=activation_3,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #卷积
        Conv_4=tf.nn.conv2d(name='conv_4',
                            input=pooling_3,
                            filter=kernel_4,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_4=tf.add(Conv_4,Bais_4)
        activation_4=tf.nn.relu(Layer_4)
        #池化
        pooling_4=tf.nn.pool(name='pooling_4',
                            input=activation_4,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #卷积
        Conv_5=tf.nn.conv2d(name='conv_5',
                            input=pooling_4,
                            filter=kernel_5,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_5=tf.add(Conv_5,Bais_8)
        activation_5=tf.nn.relu(Layer_5)
        #池化
        pooling_5=tf.nn.pool(name='pooling_5',
                            input=activation_5,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #5*5*128
        pooling_5=tf.reshape(pooling_5,[-1,5*5*128])
        #dense
        dense_1=tf.add(tf.matmul(pooling_5,w_1),Bais_5)
        activation_5=tf.nn.relu(dense_1)
        drop_out_1=tf.nn.dropout(name='drop_out_1',
                                keep_prob=__keepprob[0],x=activation_5)
        #dense
        dense_2=tf.add(tf.matmul(activation_5,w_2),Bais_6)
        activation_6=tf.nn.relu(dense_2)
        drop_out_2=tf.nn.dropout(name='drop_out_2',
                                keep_prob=__keepprob[1],x=activation_6)
        #dense
        dense_3=tf.add(tf.matmul(drop_out_2,w_3),Bais_7)
        #soft_max
        output=tf.nn.softmax(dense_3)
   #定义训练次数
    Rate=0.0001
    epoch=100
    cnn_output=output
    #定义交叉熵损失函数
    cross_entropy=tf.losses.sparse_softmax_cross_entropy(labels=__train_label,logits=cnn_output)
    train_step=tf.train.AdamOptimizer(learning_rate=Rate).minimize(cross_entropy)
    Accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(cnn_output, 1), tf.int32),__train_label),tf.float32))
    #保存最近一次模型与精度
    saver=tf.train.Saver(max_to_keep=1)
    f=open('ckpt/Accuracy.txt','w')
    #读入训练样本
    with tf.device(cpu):
        train_data_p,train_label_p,test_data_p,test_label_p=__load_data(Dir='picture',Rate=0.8)
    #会话GPU设置
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:
        #保存计算图
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range (epoch):
            #平均准确率与最大准确率
            accuracy_total=0
            accuracy_max=0
            #平均损失
            loss_total=0
            #batch总数
            n_batch=0
            #batch大小为20,训练模型
            for batch_train_data,batch_train_label in __minibatches(inputs=train_data_p,targets=train_label_p,batch_size=10,shuffle=True):
                #图像增强
                batch_train_data_enhanced,batch_train_label_enhanced=data_enhancement(batch_train_data,batch_train_label)
                with tf.device(gpu):
                    a=time.time()
                    [_,train_accuracy,loss]=sess.run([train_step,Accuracy,cross_entropy],feed_dict = {__train_data:batch_train_data_enhanced,__train_label: batch_train_label_enhanced,__keepprob:[0.8,0.5]})
                    n_batch=n_batch+1
                    loss_total=loss_total+loss
                    accuracy_total=accuracy_total+train_accuracy
                    b=time.time()
                    print('time:'+str(b-a))
            if i % 1 == 0:
                accuracy_total=accuracy_total/n_batch
                loss_total=loss_total/n_batch
                with tf.device(gpu):
                    test_accuracy = Accuracy.eval(feed_dict = {__train_data: test_data_p,__train_label:test_label_p,__keepprob:[1,1]})
                #记录最高准确率,保存准确率最高的模型并记录准确率
                f.write(str(i+1)+', val_acc: '+str(test_accuracy)+'\n')
                if accuracy_max<=accuracy_total:
                    accuracy_max=accuracy_total
                print('setp {},the train accuracy: {}'.format(i, accuracy_total))
                print('--------the test accuracy :{}'.format(test_accuracy))
                if accuracy_max>=0.95:
                    saver.save(sess,'ckpt/CNN_faces_recognition.ckpt',global_step=1) 
                    break
            #准确率大于0.99则保存模型结束训练
    f.close()
    sess.close()
#定义集GPU加速成学习基模型
def train_weak_model_1(acc):
    cpu='/cpu:0'
    gpu='/gpu:0'
    with tf.device(cpu):
        #定义占位符与变量
        __keepprob=tf.placeholder(name='drop_out',dtype=tf.float32,shape=[2])
        __train_label=tf.placeholder(name='train_label',dtype=tf.int32,shape=[None,])
        __train_data=tf.placeholder(name='train_data',dtype=tf.float32,shape=[None,size,size,1])
        kernel_1=tf.get_variable(name='kernel_1_1',
                                shape=[3,3,1,16],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_1=tf.get_variable(name='Bais_1_1',
                                shape=[1,1,1,16],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        w_1=tf.get_variable(name='w_1_1',
                            shape=[32*32*16,1024],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        w_2=tf.get_variable(name='w_1_2',
                            shape=[1024,2],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        Bais_2=tf.get_variable(name='Bais_1_2',
                                shape=[1,1024],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_3=tf.get_variable(name='Bais_1_3',
                                shape=[1,2],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
    with tf.device(cpu):   
        #卷积
        Conv_1=tf.nn.conv2d(name='conv_1_1',
                            input=__train_data,
                            filter=kernel_1,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_1=tf.add(Conv_1,Bais_1)
        activation_1=tf.nn.relu(Layer_1)
        #池化
        pooling_1=tf.nn.pool(name='pooling_1_1',
                            input=activation_1,
                            padding='SAME',
                            strides=[4,4],
                            window_shape=[4,4],
                            pooling_type='MAX')
        #10*10*64
        pooling_1=tf.reshape(pooling_1,[-1,32*32*16])
        #dense
        dense_1=tf.add(tf.matmul(pooling_1,w_1),Bais_2)
        activation_2=tf.nn.relu(dense_1)
        drop_out_1=tf.nn.dropout(name='drop_out_1_1',
                                keep_prob=__keepprob[0],x=activation_2)
        #dense
        dense_2=tf.add(tf.matmul(drop_out_1,w_2),Bais_3)
        #soft_max
        output=tf.nn.softmax(dense_2)
   #定义训练次数
    Rate=0.0001
    epoch=100
    cnn_output=output
    #定义交叉熵损失函数
    cross_entropy=tf.losses.sparse_softmax_cross_entropy(labels=__train_label,logits=cnn_output)
    train_step=tf.train.AdamOptimizer(learning_rate=Rate).minimize(cross_entropy)
    Accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(cnn_output, 1), tf.int32),__train_label),tf.float32))
    #保存最近一次模型与精度
    saver=tf.train.Saver(max_to_keep=1)
    f=open('ckpt/Accuracy.txt','w')
    #读入训练样本
    with tf.device(cpu):
        train_data_p,train_label_p,test_data_p,test_label_p=__load_data(Dir='picture',Rate=0.8)
    #会话GPU设置
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:
        #保存计算图
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range (epoch):
            #平均准确率与最大准确率
            accuracy_total=0
            accuracy_max=0
            #平均损失
            loss_total=0
            #batch总数
            n_batch=0
            #batch大小为20,训练模型
            for batch_train_data,batch_train_label in __minibatches(inputs=train_data_p,targets=train_label_p,batch_size=20,shuffle=True):
                #图像增强
                batch_train_data_enhanced,batch_train_label_enhanced=data_enhancement(batch_train_data,batch_train_label)
                with tf.device(gpu):
                    a=time.time()
                    [_,train_accuracy,loss]=sess.run([train_step,Accuracy,cross_entropy],feed_dict = {__train_data:batch_train_data_enhanced,__train_label: batch_train_label_enhanced,__keepprob:[0.7,0.5]})
                    n_batch=n_batch+1
                    loss_total=loss_total+loss
                    accuracy_total=accuracy_total+train_accuracy
                    b=time.time()
                    print('time_1:'+str(b-a))
            if i % 1 == 0:
                accuracy_total=accuracy_total/n_batch
                loss_total=loss_total/n_batch
                with tf.device(gpu):
                    test_accuracy = Accuracy.eval(feed_dict = {__train_data: test_data_p,__train_label:test_label_p,__keepprob:[1,1]})
                #记录最高准确率,保存准确率最高的模型并记录准确率
                f.write(str(i+1)+', val_acc: '+str(test_accuracy)+'\n')
                if accuracy_max<=accuracy_total:
                    accuracy_max=accuracy_total
                print('setp {},the train accuracy: {}'.format(i, accuracy_total))
                print('--------the test accuracy :{}'.format(test_accuracy))
                if accuracy_max>=acc:
                    saver.save(sess,'ckpt/1/CNN_faces_recognition.ckpt',global_step=1) 
                    break
            #准确率大于0.99则保存模型结束训练
    f.close()
    sess.close()
def train_weak_model_2(acc):
    cpu='/cpu:0'
    gpu='/gpu:0'
    with tf.device(cpu):
        #定义占位符与变量
        __keepprob=tf.placeholder(name='drop_out',dtype=tf.float32,shape=[2])
        __train_label=tf.placeholder(name='train_label',dtype=tf.int32,shape=[None,])
        __train_data=tf.placeholder(name='train_data',dtype=tf.float32,shape=[None,size,size,1])
        kernel_1=tf.get_variable(name='kernel_2_1',
                                shape=[3,3,1,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_1=tf.get_variable(name='Bais_2_1',
                                shape=[1,1,1,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        w_1=tf.get_variable(name='w_2_1',
                            shape=[32*32*32,1024],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        w_2=tf.get_variable(name='w_2_2',
                            shape=[1024,2],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        Bais_2=tf.get_variable(name='Bais_2_2',
                                shape=[1,1024],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_3=tf.get_variable(name='Bais_2_3',
                                shape=[1,2],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
    with tf.device(cpu):   
        #卷积
        Conv_1=tf.nn.conv2d(name='conv_2_1',
                            input=__train_data,
                            filter=kernel_1,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_1=tf.add(Conv_1,Bais_1)
        activation_1=tf.nn.relu(Layer_1)
        #池化
        pooling_1=tf.nn.pool(name='pooling_2_1',
                            input=activation_1,
                            padding='SAME',
                            strides=[4,4],
                            window_shape=[4,4],
                            pooling_type='MAX')
        #10*10*64
        pooling_1=tf.reshape(pooling_1,[-1,32*32*32])
        #dense
        dense_1=tf.add(tf.matmul(pooling_1,w_1),Bais_2)
        activation_2=tf.nn.relu(dense_1)
        drop_out_1=tf.nn.dropout(name='drop_out_2_1',
                                keep_prob=__keepprob[0],x=activation_2)
        #dense
        dense_2=tf.add(tf.matmul(drop_out_1,w_2),Bais_3)
        #soft_max
        output=tf.nn.softmax(dense_2)
   #定义训练次数
    Rate=0.0001
    epoch=100
    cnn_output=output
    #定义交叉熵损失函数
    cross_entropy=tf.losses.sparse_softmax_cross_entropy(labels=__train_label,logits=cnn_output)
    train_step=tf.train.AdamOptimizer(learning_rate=Rate).minimize(cross_entropy)
    Accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(cnn_output, 1), tf.int32),__train_label),tf.float32))
    #保存最近一次模型与精度
    saver=tf.train.Saver(max_to_keep=1)
    f=open('ckpt/2/Accuracy.txt','w')
    #读入训练样本
    with tf.device(cpu):
        train_data_p,train_label_p,test_data_p,test_label_p=__load_data(Dir='picture',Rate=0.8)
    #会话GPU设置
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:
        #保存计算图
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range (epoch):
            #平均准确率与最大准确率
            accuracy_total=0
            accuracy_max=0
            #平均损失
            loss_total=0
            #batch总数
            n_batch=0
            #batch大小为20,训练模型
            for batch_train_data,batch_train_label in __minibatches(inputs=train_data_p,targets=train_label_p,batch_size=20,shuffle=True):
                #图像增强
                batch_train_data_enhanced,batch_train_label_enhanced=data_enhancement(batch_train_data,batch_train_label)
                with tf.device(gpu):
                    a=time.time()
                    [_,train_accuracy,loss]=sess.run([train_step,Accuracy,cross_entropy],feed_dict = {__train_data:batch_train_data_enhanced,__train_label: batch_train_label_enhanced,__keepprob:[0.7,0.5]})
                    n_batch=n_batch+1
                    loss_total=loss_total+loss
                    accuracy_total=accuracy_total+train_accuracy
                    b=time.time()
                    print('time_2:'+str(b-a))
            if i % 1 == 0:
                accuracy_total=accuracy_total/n_batch
                loss_total=loss_total/n_batch
                with tf.device(gpu):
                    test_accuracy = Accuracy.eval(feed_dict = {__train_data: test_data_p,__train_label:test_label_p,__keepprob:[1,1]})
                #记录最高准确率,保存准确率最高的模型并记录准确率
                f.write(str(i+1)+', val_acc: '+str(test_accuracy)+'\n')
                if accuracy_max<=accuracy_total:
                    accuracy_max=accuracy_total
                print('setp {},the train accuracy: {}'.format(i, accuracy_total))
                print('--------the test accuracy :{}'.format(test_accuracy))
                if accuracy_max>=acc:
                    saver.save(sess,'ckpt/2/CNN_faces_recognition.ckpt',global_step=1) 
                    break
            #准确率大于0.99则保存模型结束训练
    f.close()
    sess.close()
def train_weak_model_3(acc):
    cpu='/cpu:0'
    gpu='/gpu:0'
    with tf.device(cpu):
        #定义占位符与变量
        __keepprob=tf.placeholder(name='drop_out',dtype=tf.float32,shape=[2])
        __train_label=tf.placeholder(name='train_label',dtype=tf.int32,shape=[None,])
        __train_data=tf.placeholder(name='train_data',dtype=tf.float32,shape=[None,size,size,1])
        kernel_1=tf.get_variable(name='kernel_3_1',
                                shape=[3,3,1,16],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_1=tf.get_variable(name='Bais_3_1',
                                shape=[1,1,1,16],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        kernel_2=tf.get_variable(name='kernel_3_2',
                                shape=[3,3,16,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01)) 
        Bais_2=tf.get_variable(name='Bais_3_2',
                                shape=[1,1,1,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        kernel_3=tf.get_variable(name='kernel_3_3',
                                shape=[3,3,32,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_3=tf.get_variable(name='Bais_3_3',
                                shape=[1,1,1,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        w_1=tf.get_variable(name='w_3_1',
                            shape=[16*16*32,1024],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        w_2=tf.get_variable(name='w_3_2',
                            shape=[1024,2],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        Bais_4=tf.get_variable(name='Bais_3_4',
                                shape=[1,1024],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_5=tf.get_variable(name='Bais_3_5',
                                shape=[1,2],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
    with tf.device(cpu):   
        #卷积
        Conv_1=tf.nn.conv2d(name='conv_3_1',
                            input=__train_data,
                            filter=kernel_1,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_1=tf.add(Conv_1,Bais_1)
        activation_1=tf.nn.relu(Layer_1)
        #池化
        pooling_1=tf.nn.pool(name='pooling_3_1',
                            input=activation_1,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #卷积
        Conv_2=tf.nn.conv2d(name='conv_3_2',
                            input=pooling_1,
                            filter=kernel_2,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_2=tf.add(Conv_2,Bais_2)
        activation_2=tf.nn.relu(Layer_2)
        #池化
        pooling_2=tf.nn.pool(name='pooling_3_2',
                            input=activation_2,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #卷积
        Conv_3=tf.nn.conv2d(name='conv_3_3',
                            input=pooling_2,
                            filter=kernel_3,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_3=tf.add(Conv_3,Bais_3)
        activation_3=tf.nn.relu(Layer_3)
        #池化
        pooling_3=tf.nn.pool(name='pooling_3_3',
                            input=activation_3,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #10*10*64
        pooling_3=tf.reshape(pooling_3,[-1,16*16*32])
        #dense
        dense_1=tf.add(tf.matmul(pooling_3,w_1),Bais_4)
        activation_4=tf.nn.relu(dense_1)
        drop_out_1=tf.nn.dropout(name='drop_out_3_1',
                                keep_prob=__keepprob[0],x=activation_4)
        #dense
        dense_2=tf.add(tf.matmul(drop_out_1,w_2),Bais_5)
        #soft_max
        output=tf.nn.softmax(dense_2)
   #定义训练次数
    Rate=0.0001
    epoch=100
    cnn_output=output
    #定义交叉熵损失函数
    cross_entropy=tf.losses.sparse_softmax_cross_entropy(labels=__train_label,logits=cnn_output)
    train_step=tf.train.AdamOptimizer(learning_rate=Rate).minimize(cross_entropy)
    Accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(cnn_output, 1), tf.int32),__train_label),tf.float32))
    #保存最近一次模型与精度
    saver=tf.train.Saver(max_to_keep=1)
    f=open('ckpt/3/Accuracy.txt','w')
    #读入训练样本
    with tf.device(cpu):
        train_data_p,train_label_p,test_data_p,test_label_p=__load_data(Dir='picture',Rate=0.8)
    #会话GPU设置
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:
        #保存计算图
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range (epoch):
            #平均准确率与最大准确率
            accuracy_total=0
            accuracy_max=0
            #平均损失
            loss_total=0
            #batch总数
            n_batch=0
            #batch大小为20,训练模型
            for batch_train_data,batch_train_label in __minibatches(inputs=train_data_p,targets=train_label_p,batch_size=20,shuffle=True):
                #图像增强
                batch_train_data_enhanced,batch_train_label_enhanced=data_enhancement(batch_train_data,batch_train_label)
                with tf.device(gpu):
                    a=time.time()
                    [_,train_accuracy,loss]=sess.run([train_step,Accuracy,cross_entropy],feed_dict = {__train_data:batch_train_data_enhanced,__train_label: batch_train_label_enhanced,__keepprob:[0.7,0.5]})
                    n_batch=n_batch+1
                    loss_total=loss_total+loss
                    accuracy_total=accuracy_total+train_accuracy
                    b=time.time()
                    print('time_3:'+str(b-a))
            if i % 1 == 0:
                accuracy_total=accuracy_total/n_batch
                loss_total=loss_total/n_batch
                with tf.device(gpu):
                    test_accuracy = Accuracy.eval(feed_dict = {__train_data: test_data_p,__train_label:test_label_p,__keepprob:[1,1]})
                #记录最高准确率,保存准确率最高的模型并记录准确率
                f.write(str(i+1)+', val_acc: '+str(test_accuracy)+'\n')
                if accuracy_max<=accuracy_total:
                    accuracy_max=accuracy_total
                print('setp {},the train accuracy: {}'.format(i, accuracy_total))
                print('--------the test accuracy :{}'.format(test_accuracy))
                if accuracy_max>=acc:
                    saver.save(sess,'ckpt/3/CNN_faces_recognition.ckpt',global_step=1) 
                    break
            #准确率大于0.99则保存模型结束训练
    f.close()
    sess.close()
def train_weak_model_4(acc):
    cpu='/cpu:0'
    gpu='/gpu:0'
    with tf.device(cpu):
        #定义占位符与变量
        __keepprob=tf.placeholder(name='drop_out',dtype=tf.float32,shape=[2])
        __train_label=tf.placeholder(name='train_label',dtype=tf.int32,shape=[None,])
        __train_data=tf.placeholder(name='train_data',dtype=tf.float32,shape=[None,size,size,1])
        kernel_1=tf.get_variable(name='kernel_4_1',
                                shape=[3,3,1,16],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_1=tf.get_variable(name='Bais_4_1',
                                shape=[1,1,1,16],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        kernel_2=tf.get_variable(name='kernel_4_2',
                                shape=[3,3,16,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_2=tf.get_variable(name='Bais_4_2',
                                shape=[1,1,1,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        kernel_3=tf.get_variable(name='kernel_4_3',
                                shape=[3,3,32,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_3=tf.get_variable(name='Bais_4_3',
                                shape=[1,1,1,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        kernel_4=tf.get_variable(name='kernel_4_4',
                                shape=[3,3,32,64],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_4=tf.get_variable(name='Bais_4_4',
                                shape=[1,1,1,64],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        w_1=tf.get_variable(name='w_4_1',
                            shape=[8*8*64,1024],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        w_2=tf.get_variable(name='w_4_2',
                            shape=[1024,2],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        Bais_5=tf.get_variable(name='Bais_4_5',
                                shape=[1,1024],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_6=tf.get_variable(name='Bais_4_6',
                                shape=[1,2],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
    with tf.device(cpu):   
        #卷积
        Conv_1=tf.nn.conv2d(name='conv_4_1',
                            input=__train_data,
                            filter=kernel_1,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_1=tf.add(Conv_1,Bais_1)
        activation_1=tf.nn.relu(Layer_1)
        #池化
        pooling_1=tf.nn.pool(name='pooling_4_1',
                            input=activation_1,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #卷积
        Conv_2=tf.nn.conv2d(name='conv_4_2',
                            input=pooling_1,
                            filter=kernel_2,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_2=tf.add(Conv_2,Bais_2)
        activation_2=tf.nn.relu(Layer_2)
        #池化
        pooling_2=tf.nn.pool(name='pooling_4_2',
                            input=activation_2,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #卷积
        Conv_3=tf.nn.conv2d(name='conv_4_3',
                            input=pooling_2,
                            filter=kernel_3,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_3=tf.add(Conv_3,Bais_3)
        activation_3=tf.nn.relu(Layer_3)
        #池化
        pooling_3=tf.nn.pool(name='pooling_4_3',
                            input=activation_3,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #卷积
        Conv_4=tf.nn.conv2d(name='conv_4_4',
                            input=pooling_3,
                            filter=kernel_4,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_4=tf.add(Conv_4,Bais_4)
        activation_4=tf.nn.relu(Layer_4)
        #池化
        pooling_4=tf.nn.pool(name='pooling_4_4',
                            input=activation_4,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #10*10*64
        pooling_4=tf.reshape(pooling_4,[-1,8*8*64])
        #dense
        dense_1=tf.add(tf.matmul(pooling_4,w_1),Bais_5)
        activation_5=tf.nn.relu(dense_1)
        drop_out_1=tf.nn.dropout(name='drop_out_4_1',
                                keep_prob=__keepprob[0],x=activation_5)
        #dense
        dense_2=tf.add(tf.matmul(drop_out_1,w_2),Bais_6)
        #soft_max
        output=tf.nn.softmax(dense_2)
   #定义训练次数
    Rate=0.0001
    epoch=100
    cnn_output=output
    #定义交叉熵损失函数
    cross_entropy=tf.losses.sparse_softmax_cross_entropy(labels=__train_label,logits=cnn_output)
    train_step=tf.train.AdamOptimizer(learning_rate=Rate).minimize(cross_entropy)
    Accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(cnn_output, 1), tf.int32),__train_label),tf.float32))
    #保存最近一次模型与精度
    saver=tf.train.Saver(max_to_keep=1)
    f=open('ckpt/4/Accuracy.txt','w')
    #读入训练样本
    with tf.device(cpu):
        train_data_p,train_label_p,test_data_p,test_label_p=__load_data(Dir='picture',Rate=0.8)
    #会话GPU设置
    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:
        #保存计算图
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range (epoch):
            #平均准确率与最大准确率
            accuracy_total=0
            accuracy_max=0
            #平均损失
            loss_total=0
            #batch总数
            n_batch=0
            #batch大小为20,训练模型
            for batch_train_data,batch_train_label in __minibatches(inputs=train_data_p,targets=train_label_p,batch_size=20,shuffle=True):
                #图像增强
                batch_train_data_enhanced,batch_train_label_enhanced=data_enhancement(batch_train_data,batch_train_label)
                with tf.device(gpu):
                    a=time.time()
                    [_,train_accuracy,loss]=sess.run([train_step,Accuracy,cross_entropy],feed_dict = {__train_data:batch_train_data_enhanced,__train_label: batch_train_label_enhanced,__keepprob:[0.7,0.5]})
                    n_batch=n_batch+1
                    loss_total=loss_total+loss
                    accuracy_total=accuracy_total+train_accuracy
                    b=time.time()
                    print('time_4:'+str(b-a))
            if i % 1 == 0:
                accuracy_total=accuracy_total/n_batch
                loss_total=loss_total/n_batch
                with tf.device(gpu):
                    test_accuracy = Accuracy.eval(feed_dict = {__train_data: test_data_p,__train_label:test_label_p,__keepprob:[1,1]})
                #记录最高准确率,保存准确率最高的模型并记录准确率
                f.write(str(i+1)+', val_acc: '+str(test_accuracy)+'\n')
                if accuracy_max<=accuracy_total:
                    accuracy_max=accuracy_total
                print('setp {},the train accuracy: {}'.format(i, accuracy_total))
                print('--------the test accuracy :{}'.format(test_accuracy))
                if accuracy_max>=acc:
                    saver.save(sess,'ckpt/4/CNN_faces_recognition.ckpt',global_step=1) 
                    break
            #准确率大于0.99则保存模型结束训练
    f.close()
    sess.close()
def train_weak_model_5(acc):
    cpu='/cpu:0'
    gpu='/gpu:0'
    with tf.device(cpu):
        #定义占位符与变量
        __keepprob=tf.placeholder(name='drop_out',dtype=tf.float32,shape=[2])
        __train_label=tf.placeholder(name='train_label',dtype=tf.int32,shape=[None,])
        __train_data=tf.placeholder(name='train_data',dtype=tf.float32,shape=[None,size,size,1])
        kernel_1=tf.get_variable(name='kernel_5_1',
                                shape=[3,3,1,16],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_1=tf.get_variable(name='Bais_5_1',
                                shape=[1,1,1,16],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        kernel_2=tf.get_variable(name='kernel_5_2',
                                shape=[3,3,16,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_2=tf.get_variable(name='Bais_5_2',
                                shape=[1,1,1,32],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        w_1=tf.get_variable(name='w_5_1',
                            shape=[16*16*32,1024],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        w_2=tf.get_variable(name='w_5_2',
                            shape=[1024,2],
                            dtype=np.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(0.003))
        Bais_3=tf.get_variable(name='Bais_5_3',
                                shape=[1,1024],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        Bais_4=tf.get_variable(name='Bais_5_4',
                                shape=[1,2],
                                dtype=np.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
    with tf.device(cpu):   
        #卷积
        Conv_1=tf.nn.conv2d(name='conv_5_1',
                            input=__train_data,
                            filter=kernel_1,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_1=tf.add(Conv_1,Bais_1)
        activation_1=tf.nn.relu(Layer_1)
        #池化
        pooling_1=tf.nn.pool(name='pooling_5_1',
                            input=activation_1,
                            padding='SAME',
                            strides=[2,2],
                            window_shape=[2,2],
                            pooling_type='MAX')
        #卷积
        Conv_2=tf.nn.conv2d(name='conv_5_1',
                            input=pooling_1,
                            filter=kernel_2,
                            padding='SAME',
                            strides=[1,1,1,1],
                            use_cudnn_on_gpu=True)
        Layer_2=tf.add(Conv_2,Bais_2)
        activation_2=tf.nn.relu(Layer_2)
        #池化
        pooling_2=tf.nn.pool(name='pooling_5_1',
                            input=activation_2,
                            padding='SAME',
                            strides=[4,4],
                            window_shape=[4,4],
                            pooling_type='MAX')
        #10*10*64
        pooling_2=tf.reshape(pooling_2,[-1,16*16*32])
        #dense
        dense_1=tf.add(tf.matmul(pooling_2,w_1),Bais_3)
        activation_2=tf.nn.relu(dense_1)
        drop_out_1=tf.nn.dropout(name='drop_out_5_1',
                                keep_prob=__keepprob[0],x=activation_2)
        #dense
        dense_2=tf.add(tf.matmul(drop_out_1,w_2),Bais_4)
        #soft_max
        output=tf.nn.softmax(dense_2)
   #定义训练次数
    Rate=0.0001
    epoch=100
    cnn_output=output
    #定义交叉熵损失函数
    cross_entropy=tf.losses.sparse_softmax_cross_entropy(labels=__train_label,logits=cnn_output)
    train_step=tf.train.AdamOptimizer(learning_rate=Rate).minimize(cross_entropy)
    Accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(cnn_output, 1), tf.int32),__train_label),tf.float32))
    #保存最近一次模型与精度
    saver=tf.train.Saver(max_to_keep=1)
    f=open('ckpt/5/Accuracy.txt','w')
    #读入训练样本
    with tf.device(cpu):
        train_data_p,train_label_p,test_data_p,test_label_p=__load_data(Dir='picture',Rate=0.8)
    #会话GPU设置
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:
        #保存计算图
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range (epoch):
            #平均准确率与最大准确率
            accuracy_total=0
            accuracy_max=0
            #平均损失
            loss_total=0
            #batch总数
            n_batch=0
            #batch大小为20,训练模型
            for batch_train_data,batch_train_label in __minibatches(inputs=train_data_p,targets=train_label_p,batch_size=20,shuffle=True):
                #图像增强
                batch_train_data_enhanced,batch_train_label_enhanced=data_enhancement(batch_train_data,batch_train_label)
                with tf.device(gpu):
                    a=time.time()
                    [_,train_accuracy,loss]=sess.run([train_step,Accuracy,cross_entropy],feed_dict = {__train_data:batch_train_data_enhanced,__train_label: batch_train_label_enhanced,__keepprob:[0.7,0.5]})
                    n_batch=n_batch+1
                    loss_total=loss_total+loss
                    accuracy_total=accuracy_total+train_accuracy
                    b=time.time()
                    print('time_2:'+str(b-a))
            if i % 1 == 0:
                accuracy_total=accuracy_total/n_batch
                loss_total=loss_total/n_batch
                with tf.device(gpu):
                    test_accuracy = Accuracy.eval(feed_dict = {__train_data: test_data_p,__train_label:test_label_p,__keepprob:[1,1]})
                #记录最高准确率,保存准确率最高的模型并记录准确率
                f.write(str(i+1)+', val_acc: '+str(test_accuracy)+'\n')
                if accuracy_max<=accuracy_total:
                    accuracy_max=accuracy_total
                print('setp {},the train accuracy: {}'.format(i, accuracy_total))
                print('--------the test accuracy :{}'.format(test_accuracy))
                if accuracy_max>=acc:
                    saver.save(sess,'ckpt/5/CNN_faces_recognition.ckpt',global_step=1) 
                    break
            #准确率大于0.99则保存模型结束训练
    f.close()
    sess.close()
#定义外部调用函数,picture为单帧图像,sess为会话句柄

def recognize(flag,src):
    if flag==1:
        cap=cv2.VideoCapture(0)
        with tf.Session() as sess:
            #建立模型框架   
            sess.run(tf.global_variables_initializer()) 
            saver = tf.train.import_meta_graph('ckpt/CNN_faces_recognition.ckpt-1.meta')
            saver.restore(sess,tf.train.latest_checkpoint('ckpt/'))
            graph = tf.get_default_graph()
            #tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
            x_=graph.get_tensor_by_name('train_data:0')
            drop_=graph.get_tensor_by_name('drop_out:0')
            cnn_output=graph.get_tensor_by_name('Softmax:0')
            #获取Haar分类器
            classfier=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
            while cap.isOpened() :
                sucess,frame = cap.read()
                #成功开启摄像头后捕获视频
                if sucess :
                    ##当前帧预处理
                    #灰度化
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    ##检测人脸
                    faces=classfier.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3,minSize=(150,150))
                    #如果检测到人脸则截取并输出
                    if len(faces)>0 :
                        for face in faces :
                            x,y,w,h = face
                            #标识出人脸区域
                            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                            #人脸截取
                            picture=frame[y:y+h,x:x+w]
                            #picture=frame[y:y+h,x:x+w]
                            #灰度化
                            gray=cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
                            #图像归一化预处理
                            gray=cv2.resize(gray,(size,size))
                            gray=np.reshape(gray,[size,size,1])
                            #直方图均衡
                            gray=cv2.equalizeHist(gray)
                            with tf.device('/gpu:0'):
                                data_norm = np.zeros(gray.shape, dtype=np.float32)
                                cv2.normalize(gray, data_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                data_norm=np.reshape(data_norm,[1,size,size,1])
                                softmax_=tf.nn.softmax(sess.run(cnn_output,feed_dict={x_:data_norm,drop_:[1,1]}))
                                result_label=tf.argmax(softmax_,1)
                                result=result_label.eval()
                                #str_=''
                                #str_=__list[result[0]]+str(softmax_[0][result[0]].eval())
                                str_=__list[result[0]]
                            #标出识别结果
                            cv2.putText(frame,str_,(x-50,y-30),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)  
                    cv2.imshow('frame',frame)
                    if cv2.waitKey(1)==27:
                        break
    elif flag==0:
         with tf.Session() as sess:
            #建立模型框架   
            sess.run(tf.global_variables_initializer()) 
            saver = tf.train.import_meta_graph('ckpt/CNN_faces_recognition.ckpt-1.meta')
            saver.restore(sess,tf.train.latest_checkpoint('ckpt/'))
            graph = tf.get_default_graph()
            x_=graph.get_tensor_by_name('graph/train_data:0')
            drop_=graph.get_tensor_by_name('graph/drop_out:0')
            cnn_output=graph.get_tensor_by_name('graph/output/BiasAdd:0')
            #获取Haar分类器
            classfier=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
            img=cv2.imread(src,1)
            #灰度化
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #图像归一化预处理
            gray=cv2.resize(gray,(size,size))
            #直方图均衡
            gray=cv2.equalizeHist(gray)
            data_norm = np.zeros(gray.shape, dtype=np.float32)
            cv2.normalize(gray, data_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            data_norm=np.reshape(data_norm,[1,size,size,1])
            softmax_=tf.nn.softmax(sess.run(cnn_output,feed_dict={x_:data_norm,drop_:False}))
            result_label=tf.argmax(softmax_,1)
            result=result_label.eval()
            print(__list[result[0]])
    '''
    elif flag==2:
        cap=cv2.VideoCapture(0)      
        g1=tf.Graph()
        g2=tf.Graph()
        g3=tf.Graph()
        g4=tf.Graph()
        g5=tf.Graph()
        sess_1=tf.Session(graph=g1)
        sess_2=tf.Session(graph=g2)
        sess_3=tf.Session(graph=g3)
        sess_4=tf.Session(graph=g4)
        sess_5=tf.Session(graph=g5)
        with g1.as_default():
            #建立模型框架   
            sess_1.run(tf.global_variables_initializer()) 
            saver = tf.train.import_meta_graph('ckpt/1/CNN_faces_recognition.ckpt-1.meta')
            saver.restore(sess_1,tf.train.latest_checkpoint('ckpt/1/'))
        with g2.as_default():
            #建立模型框架   
            sess_2.run(tf.global_variables_initializer()) 
            saver = tf.train.import_meta_graph('ckpt/2/CNN_faces_recognition.ckpt-1.meta')
            saver.restore(sess_2,tf.train.latest_checkpoint('ckpt/2/'))
        with g3.as_default():
            #建立模型框架   
            sess_3.run(tf.global_variables_initializer()) 
            saver = tf.train.import_meta_graph('ckpt/3/CNN_faces_recognition.ckpt-1.meta')
            saver.restore(sess_3,tf.train.latest_checkpoint('ckpt/3/'))
        with g4.as_default():
            #建立模型框架   
            sess_4.run(tf.global_variables_initializer()) 
            saver = tf.train.import_meta_graph('ckpt/4/CNN_faces_recognition.ckpt-1.meta')
            saver.restore(sess_4,tf.train.latest_checkpoint('ckpt/4/'))
        with g5.as_default():
            #建立模型框架   
            sess_5.run(tf.global_variables_initializer()) 
            saver = tf.train.import_meta_graph('ckpt/5/CNN_faces_recognition.ckpt-1.meta')
            saver.restore(sess_5,tf.train.latest_checkpoint('ckpt/5/'))
        x_=[]
        keep_prob=[]
        cnn_output=[]
        for i in [g1,g2,g3,g4,g5]:
            #tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
            x_.append(i.get_tensor_by_name('train_data:0'))
            keep_prob.append(i.get_tensor_by_name('drop_out:0'))
            cnn_output.append(i.get_tensor_by_name('Softmax:0'))
        #获取Haar分类器
        classfier=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        while cap.isOpened() :
            sucess,frame = cap.read()
            #成功开启摄像头后捕获视频
            if sucess :
                ##当前帧预处理
                #灰度化
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                ##检测人脸
                faces=classfier.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3,minSize=(50,50))
                #如果检测到人脸则截取并输出
                if len(faces)>0 :
                    for face in faces :
                        x,y,w,h = face
                        #标识出人脸区域
                        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                        #人脸截取
                        picture=frame[y:y+h,x:x+w]
                        #picture=frame[y:y+h,x:x+w]
                        #灰度化
                        gray=cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
                        #图像归一化预处理
                        gray=cv2.resize(gray,(size,size))
                        #直方图均衡
                        #gray=cv2.equalizeHist(gray)
                        with tf.device('/gpu:0'):
                            data_norm = np.zeros(gray.shape, dtype=np.float32)
                            cv2.normalize(gray, data_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                            data_norm=np.reshape(data_norm,[1,size,size,1])
                            softmax_=[]
                            softmax_.append(np.argmax(sess_1.run(cnn_output[0],feed_dict={x_[0]:data_norm,keep_prob[0]:[1,1]}),1)[0])
                            softmax_.append(np.argmax(sess_2.run(cnn_output[1],feed_dict={x_[1]:data_norm,keep_prob[1]:[1,1]}),1)[0])
                            softmax_.append(np.argmax(sess_3.run(cnn_output[2],feed_dict={x_[2]:data_norm,keep_prob[2]:[1,1]}),1)[0])
                            softmax_.append(np.argmax(sess_4.run(cnn_output[3],feed_dict={x_[3]:data_norm,keep_prob[3]:[1,1]}),1)[0])
                            softmax_.append(np.argmax(sess_5.run(cnn_output[4],feed_dict={x_[4]:data_norm,keep_prob[4]:[1,1]}),1)[0])
                            #投票判决
                            result_label=0
                            for i in range(5):
                                if softmax_[i]==0:
                                    result_label=result_label+1
                            if result_label>=3:
                                result=0
                            else:
                                result=1
                            #str_=__list[result[0]]+str(softmax_[0][result[0]].eval())
                            str_=__list[result]
                        #标出识别结果
                        cv2.putText(frame,str_,(x-50,y-30),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)  
                cv2.imshow('frame',frame)
                if cv2.waitKey(1)==27:
                    break 
'''
if __name__ == "__main__":
    #卷积神经网络方法
    train=1
    if train==1:
        #train_cnn_model(4)
        trian_gpu_model()
        #train_weak_model_1(0.8)
        #train_weak_model_2(0.8)
        #train_weak_model_3(0.8)
        #train_weak_model_4(0.8)
        #train_weak_model_5(0.8)
    #elif train==0:
        #recognize(1,[])

