#!/usr/bin/python
#coding=utf-8

import os
import random
import numpy as np
import cv2

# ----------------参数----------------
IMAGE_SIZE = 64
IMAGE_NUM = 200
COLOR_BLUE = (255, 0, 0)
TRAIN_IMAGE_DIR_PATH = './image/trainfaces'
DELAY_TIME_MS = 30
# ------------------------------------

def create_dir(*args):
    '''
    创建目录
    
    Parameters
    ----------
    *args : 目录的名称
    '''
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

def relight(imgsrc, alpha=1, bias=0):
    '''
    随机改变图片的亮度
    
    Parameters
    ----------
    alpha : 改变的比例
    
    bias : 随机偏移值
    
    Return
    ------
    img : 处理后的图片
    '''
    img = imgsrc.astype(float)
    img = img * alpha + bias
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)
    return img

def get_face_from_camera(dir_path, name):
    '''
    从摄像头获取脸部数据，存入dir_path中name文件夹中
    
    Parameters
    ----------
    dir_path : 存放图片的目录
    
    name : 姓名，作为二级目录
    '''
    
    # 得到保存照片的路径
    faces_dir_path = os.path.join(dir_path, name)
    create_dir(faces_dir_path)
    
    # 打开摄像头
    camera = cv2.VideoCapture(0)
    
    # 获取分类器实例
    haar = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    
    # 循环获取IMAGE_NUM张照片
    pic_index = 0;
    while True:
        print('It`s processing %s image.' % pic_index)
        
        # 获取一帧图片
        success, img = camera.read()
        
        # 将图片转化为灰度图片传给分类器检测人脸的位置
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray_img, 1.3, 5)
        
        # 将检测到的人脸部分进行处理后存储到指定目录
        for f_x, f_y, f_w, f_h in faces:
            # 获取人脸部分数据并resize到指定大小
            face = img[f_y: f_y+f_h, f_x: f_x+f_w]
            face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
            
            # 对人脸的亮度进行调随机整，有利于训练数据
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            
            # 将人脸图片进行存储
            cv2.imwrite(os.path.join(faces_dir_path, str(pic_index)+'.jpg'), face)
            pic_index += 1
            
            if pic_index == IMAGE_NUM:
                break
            # 显示人脸代表的人的姓名
            cv2.putText(img, name, (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 2)
            
            # 将人脸的部分用蓝色的框框起来(BGR而不是RGB)
            img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), COLOR_BLUE, 2)
        
        
        # 显示图片并稍作延迟
        cv2.imshow('img', img)
        cv2.waitKey(DELAY_TIME_MS) & 0xff
        
        if pic_index == IMAGE_NUM:
            break;
    
    # 释放摄像头资源
    camera.release()
    # 销毁所有窗体
    cv2.destroyAllWindows()

if __name__ == '__main__':
    name = input('Please input your name: ')
    get_face_from_camera(TRAIN_IMAGE_DIR_PATH, name)