#!/usr/bin/python
#coding=utf-8

import numpy as np
import tensorflow as tf
import tensorflow_face as tff

# ----------------参数----------------
IMAGE_SIZE = 64
BATCH_SIZE = 20
TRAIN_IMAGE_DIR_PATH = './image/trainfaces'
TEST_IMAGE_DIR_PATH = './image/testfaces'
CASCADE_CLASSIFIER = 'haarcascade_frontalface_default.xml'
CHECK_POINT_SAVE_PATH = './checkpoint/face.ckpt'
# ------------------------------------

x_data = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
y_data = tf.placeholder(tf.float32, [None, None])
keep_prob = tf.placeholder(tf.float32)

# 生成指定形状的权值
def weight_variable(shape):
    '''
    按指定shape生成随机权值
    
    Parameters
    ----------
    shape : 权值的形状
    
    Return
    ------
    随机权值
    '''
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# 生成指定形状的偏移
def bias_variable(shape):
    '''
    按指定shape生成随机偏移
    
    Parameters
    ----------
    shape : 偏移的形状
    
    Return
    ------
    随机偏移
    '''
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# 卷积运算
def conv2d(x, W):
    '''
    对x和W进行卷积运算
    
    Parameters
    ----------
    x : 原始数据
    W : 权值
    
    Return 
    ------
    卷积后的数据
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化运算
def max_pool_2x2(x):
    '''
    对x进行池化操作
    
    Parameters
    ----------
    x : 原始数据
    
    Return 
    ------
    池化后的数据
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 创建cnn网络
def create_cnn(class_num):
    '''
    创建cnn网络
    
    Parameters
    ----------
    class_num : 分类数量
    
    Return
    ------
    最后输出层的结果[train_num * class_num]
    '''
    # 第一层
    W_conv1 = weight_variable([5, 5, 3, 32])    # 卷积核大小(5, 5)， 输入通道(3)， 输出通道(32)
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_data, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 第三层
    W_conv3 = weight_variable([5, 5, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    # 全连接层1
    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # 全连接层2
    W_fc2 = weight_variable([1024, 512])
    b_fc2 = bias_variable([512])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # 输出层
    W_fc2 = weight_variable([512, class_num])
    b_fc2 = bias_variable([class_num])
    out = tf.matmul(h_fc2_drop, W_fc2) + b_fc2
    return out

# 训练cnn网络
def train_cnn(train_x, train_y, test_x, test_y, train_num, tf_savepath):
    '''
    训练cnn网络
    
    Parameters
    ----------
    train_x : 训练样本
    train_y : 训练样本的标签
    train_num : 训练次数
    '''
    # 获取cnn的输出
    out = create_cnn(train_y.shape[1])
    # 根据样本的标签值和cnn的输出定义交叉熵
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_data, logits=out))
    # 定义训练步骤
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    # 定义准确率
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # 定义训练状态保存器
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化全局变量
        sess.run(tf.global_variables_initializer())
        # 样本数
        example_num = len(train_x)
        # batch数为样本总数 // batch大小
        batch_num =  example_num // BATCH_SIZE
        # 训练train_num次
        for train_i_step in range(train_num):
            # 随机打乱训练数据和标签的顺序
            r = np.random.permutation(example_num)
            train_x_t = train_x[r, :]
            train_y_t = train_y[r, :]
            
            # 循环训练每一个batch
            for i in range(batch_num):
                batch_x = train_x_t[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                batch_y = train_y_t[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                # 获取测试数据的准确率
                sess.run(train_step, feed_dict={x_data:batch_x, y_data:batch_y, keep_prob: 0.8})
                
                train_accuracy = accuracy.eval({x_data: test_x, y_data: test_y, keep_prob: 1.0})
                print('准确率为：%g%%' % (train_accuracy * 100))
        
            # 获取测试数据的准确率
            train_accuracy = accuracy.eval({x_data: test_x, y_data: test_y, keep_prob: 1.0})
            print('第 %d 次训练的准确率为：%g%%' % (train_i_step, train_accuracy * 100))
        
        # 保存训练状态
        saver.save(sess, tf_savepath)

if __name__ == '__main__':
    path_label_pair_test, _ = tff.get_file_and_label(TEST_IMAGE_DIR_PATH)
    test_x, test_y = tff.get_data_and_label_matrix(path_label_pair_test)
    _, index_to_name = tff.get_file_and_label(TEST_IMAGE_DIR_PATH)
    # 定义训练状态保存器
    out = create_cnn(len(index_to_name))
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 恢复训练状态
        saver.restore(sess, CHECK_POINT_SAVE_PATH)
        
        train_accuracy = accuracy.eval({x_data: test_x, y_data: test_y, keep_prob: 1.0})
        print('准确率为：%g%%' % (train_accuracy * 100))
        
        
        
        
        
