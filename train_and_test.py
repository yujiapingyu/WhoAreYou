#!/usr/bin/python
# coding=utf-8

import os
import numpy as np
import tensorflow as tf
import cv2
import convolutional as my_conv

# ----------------参数----------------
IMAGE_SIZE = 64
TRAIN_NUM = 10
COLOR_BLUE = (255, 0, 0)
CHECK_POINT_SAVE_PATH = './checkpoint/face.ckpt'
TRAIN_IMAGE_DIR_PATH = './image/trainfaces'
TEST_IMAGE_DIR_PATH = './image/testfaces'
CASCADE_CLASSIFIER = 'haarcascade_frontalface_default.xml'
DELAY_TIME_MS = 30
# ------------------------------------

# 获取指定路径下的图片文件
def get_images_in_path(file_dir):
    '''
    获取指定路径file_dir下的图片文件

    Parameters
    ----------
    file_dir : 路径

    Return
    ------
    产生图片路径的生成器
    '''
    for (path, _, file_names) in os.walk(file_dir):
        for file_name in file_names:
            if file_name.endswith('.jpg'):
                yield os.path.join(path, file_name)

# 将分类数转化为独热编码
def one_hot(class_num):
    '''
    传入分类数，转化为独热编码

    Parameters
    ----------
    class_num : 分类数

    Return
    ------
    所有类的独热编码
    '''
    e = np.eye(class_num)
    return e.tolist()

# 获取指定路径下的文件和标签信息
def get_file_and_label(file_dir):
    '''
    获取指定路径file_dir路径下的文件和标签信息

    Parameters
    ----------
    file_dir : 路径

    Return
    ------
    (1) list : (path, one_hot_label)元组的列表
    (2) dict : (index: name)的字典
    '''
    # 获取指定路径下的[name, path]字典
    dir_dict = dict([[name, os.path.join(file_dir, name)] for name in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, name))])

    name_list, path_list = dir_dict.keys(), dir_dict.values()

    # 获取训练样本的类数(即有多少个不同的人)
    class_num = len(dir_dict)
    list_index = list(range(len(dir_dict)))

    return list(zip(path_list, one_hot(class_num))), dict(zip(list_index, name_list))

# 获取训练/测试数据和标签，生成数据和标签的矩阵
def get_data_and_label_matrix(path_label_pair):
    '''
    获取训练/测试数据和标签，生成数据和标签的矩阵

    Parameters
    ----------
    path_label_pair : (path, one_hot_label)元组的列表

    Return
    ------
    (1) 数据矩阵
    (2) 独热标签矩阵
    '''
    imgs = []
    labels = []
    # 遍历path_label_pair列表，获取每一张图片，生成数据和标签的矩阵
    for file_path, label in path_label_pair:
        for item in get_images_in_path(file_path):
            img = cv2.imread(item)
            imgs.append(img)
            labels.append(label)

    # 将图片的像素值转化为float32类型并归一化
    train_x = np.array(imgs).astype(np.float32) / 255.0
    train_y = np.array(labels)
    return train_x, train_y

# 从预测结果获取姓名
def get_predict_name(index_to_name, res):
    '''
    从预测结果获取姓名

    Parameters
    ----------
    index_to_name : (index: name)的字典
    res : 预测的结果

    Return
    ------
    预测对应的姓名
    '''
    index = res[1][0]
    return index_to_name[index]

# 从摄像头获取画面进行模型测试
def test_from_camera(check_point):
    '''
    从摄像头获取画面进行模型测试

    Parameters
    ----------
    check_point : 检查点的路径
    '''
    # 打开摄像头
    camera = cv2.VideoCapture(0)
    # 获取分类器实例
    haar = cv2.CascadeClassifier(CASCADE_CLASSIFIER)
    # 获取索引和名字对应的字典
    _, index_to_name = get_file_and_label(TRAIN_IMAGE_DIR_PATH)
    # 获取分类数目
    class_num = len(index_to_name)
    # 预测值
    predict = my_conv.create_cnn(class_num)
    # 定义训练状态保存器
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 恢复训练状态
        saver.restore(sess, check_point)

        # 循环检测每一帧
        while True:
            # 读取一帧
            success, img = camera.read()

            # 将图片转化为灰度图片传给分类器检测人脸的位置
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)

            # 对图片中的每一张人脸进行预测
            for f_x, f_y, f_w, f_h in faces:
                # 获取人脸部分数据并resize到指定大小
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))

                # 将人脸部分数据转化为训练数据矩阵的样式
                test_x = np.array([face])
                test_x = test_x.astype(np.float32) / 255.0

                # 得到训练结果
                res = sess.run([predict, tf.argmax(predict, 1)], feed_dict={my_conv.x_data: test_x, my_conv.keep_prob: 1.0})

                # 获取训练结果对应的人的名字
                predict_name = get_predict_name(index_to_name, res)

                # 显示名字
                cv2.putText(img, predict_name, (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 2)  

                # 将人脸的部分用蓝色的框框起来(BGR而不是RGB)
                img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), COLOR_BLUE, 2)

            # 显示图片并稍作延迟
            cv2.imshow('img', img)
            key = cv2.waitKey(DELAY_TIME_MS) & 0xff

            # 按ESC键退出
            if key == 27:
                break

    # 释放摄像头资源
    camera.release()
    # 销毁所有窗体
    cv2.destroyAllWindows()


if __name__ == '__main__':
    is_need_train = False
    # 如果存在上次的训练状态，则不需要训练
    if os.path.exists(CHECK_POINT_SAVE_PATH + '.meta') is False:
        is_need_train = True

    if is_need_train:
        # 训练数据
        path_label_pair, _ = get_file_and_label(TRAIN_IMAGE_DIR_PATH)
        train_x, train_y = get_data_and_label_matrix(path_label_pair)
        path_label_pair_test, _ = get_file_and_label(TEST_IMAGE_DIR_PATH)
        test_x, test_y = get_data_and_label_matrix(path_label_pair_test)
        my_conv.train_cnn(train_x, train_y, test_x, test_y, TRAIN_NUM, CHECK_POINT_SAVE_PATH)
    else:
        # 测试数据
        test_from_camera(CHECK_POINT_SAVE_PATH)