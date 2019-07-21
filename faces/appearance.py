from tkinter import *
import tkinter.messagebox as messagebox
import face_capture as fc
import face_recognization_train as frt
import os
 # 导入tkinter模块的所有内容
root = Tk() 
'''
训练素材存储目录为 picture/
'''
v = IntVar()
v1=StringVar()
v2=StringVar()
#操作函数
def go():
    #录入人脸
    if v.get()==1:
        if v1.get().strip():
            fc.Get_Face(0,200,'faces_0')
            frt.set_name(v1.get())
            frt.trian_gpu_model()
        else:
            messagebox.showinfo('提示','请输入人脸标识')
    elif v.get()==2:
        frt.recognize(1,[])
    elif v.get()==3:
        messagebox.askokcancel( 'c')
#操作面板
group_0=LabelFrame(root) 
Radiobutton(group_0,text='录入人脸',variable=v,value=1).pack(anchor=W)
label_1= Label(group_0,text='人脸标识 :')
e1 = Entry(group_0,textvariable=v1)
group_1=LabelFrame(root,width=37,height=10)
#text=Text(group_0,width=36,height=10)
Radiobutton(group_1,text='人脸检测',variable=v,value=2).pack(anchor=W)
group_2=LabelFrame(root) 
button_0=Button(root,text='执行',command=go)
#显示面板
group_out=Canvas (root,height=128,width=128)
group_0.pack(pady=10,padx=10) 
group_1.pack(pady=10,padx=70,fill='both')
button_0.pack(padx=150)  
#text.pack(pady=5,padx=20,side=BOTTOM,expand=YES)
label_1.pack(pady=0,padx=20,side='left')
e1.pack(pady=5,padx=10)
mainloop()
