import cv2
import numpy as np
import os
import face_recognization_train as face_r
import tensorflow as tf
#创建目录函数
def create_foleder(foldernames):
    isCreated=os.path.exists(foldernames)
    if not isCreated:
        os.makedirs(foldernames)
        print(str(foldernames)+'is created')
        return True
    else:
        print("the folder has been created before")
        return False
#前置摄像机捕获视频 
def Get_Face(cam_index,picture_num,dir_):
    dir_='picture/'+dir_
    cap = cv2.VideoCapture(cam_index)
    num=0
    #获取Haar分类器
    classfier=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    #成功开启摄像头后捕获视频
    while cap.isOpened() :
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
                for face in faces :
                    x,y,w,h = face
                    name='/%d.jpg'%(num)
                    path=dir_+name 
                    cv2.imwrite(path,frame[y:y+h,x:x+w]) 
                    #标识出人脸区域
                    frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)       
                    num=num+1
                    if num == picture_num :
                        break
            #计算采集进度并标识出图像采集进度文字
            rate = '%d'%(num*100/picture_num)
            cv2.putText(frame,'procedure:'+rate+'%',(50,30),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
            #当前类别人脸采集完毕
            if num == picture_num :
                break       
        #输出图像
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
        #如果按下ESC则退出循环
        if cv2.waitKey(1) == 27:
            break
        
    #关闭采集窗口，释放摄像头资源
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    Get_Face(0,200,'faces_0')

