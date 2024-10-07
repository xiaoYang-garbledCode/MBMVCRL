# 作者: Peach Yang
# 2023年03月12日11时27分25秒
# 作者: Peach Yang
# 2023年02月18日22时44分28秒


# knn 癫痫正确率：0.35390625
# sleepEDF SVM，直接计算准确率 0.432

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
from datetime import datetime


def finalAcc(path_features, path_labels, name, lambda1, lambda2,
             lambda3, lambda4, n_fft, hop_length, i):
    start_time = datetime.now()
    print('=============================================')
    print('正在读取sample_{} 当中，SVM需要的特征：'.format(i))
    data = np.load(path_features)
    # 读取数据标签
    labels = np.load(path_labels)
    if name == 'Epilepsy':
        # 读取特征
        features_len = 24
        final_out_channels = 128
        data = data.reshape(-1, final_out_channels * features_len)
        # 11392 - 2300 = 9092
        data_train = data[0:9092]
        data_test = data[9092:]
        labels = labels.ravel()
        data_label_train = labels[0:9092]
        data_label_test = labels[9092:]
    elif name == 'HAR':
        features_len = 18
        final_out_channels = 128
        data = data.reshape(-1, final_out_channels * features_len)
        # 11392 - 2300 = 9092
        data_train = data[0:7255]
        data_test = data[7255:]
        labels = labels.ravel()
        data_label_train = labels[0:7255]
        data_label_test = labels[7255:]
        HAR_label = np.load(path_labels)
    elif name == 'sleepEDF':
        features_len = 127
        final_out_channels = 128
        data = data.reshape(-1, final_out_channels * features_len)
        # 11392 - 2300 = 9092
        data_train = data[0:33330]
        data_test = data[33330:]
        labels = labels.ravel()
        data_label_train = labels[0:33330]
        data_label_test = labels[33330:]
    s = 10.0
    print('=============================================')
    print('开始SVM训练sample_{}： features_len * final_out_channels为：%d'.format(i) % features_len, final_out_channels)
    print('=============================================')
    SVM(data_train, data_test, data_label_train, data_label_test, s, name, lambda1, lambda2,
        lambda3, lambda4, n_fft, hop_length)
    str2 = str(i) + "： " + "features_len 和 final_out_channels为 " + str(features_len) + " " + str(final_out_channels)
    end_time = datetime.now() - start_time
    with open(name + '_svm_result.txt', 'a+') as f:
        f.write(str2 + '\n' + "运行总时间：" + '\n' + str(end_time) + '\n')
        f.close()


def SVM(epilepsy_data_train, epilepsy_data_test, epilepsy_data_label_train, epilepsy_data_label_test, C, name, lambda1,
        lambda2, lambda3, lambda4, n_fft, hop_length, i):
    # 创建SVC/Support Vector Classification/支持向量机分类器模型
    svc_model = svm.SVC(C=C)
    # 将数据拟合到SVC模型中，此处用到了标签值HAR_label_train，是有监督学习
    svc_model.fit(epilepsy_data_train, epilepsy_data_label_train)
    score = svc_model.score(epilepsy_data_test, epilepsy_data_label_test)
    y_predict = svc_model.predict(epilepsy_data_test)
    print("lambda的取值为：" + str(lambda1) + "   " + str(lambda2) + "   " + str(lambda3) + "   " + str(
        lambda4) + "  n_fft的值为：  " + str(n_fft) + "  hop_length的值为： " + str(hop_length) + "\n" + "超参数C=" + str(
        C) + "时，" + name + "的模型的正确率: %.3f" % score)
    str1 = "lambda的取值为：" + str(lambda1) + "   " + str(lambda2) + "   " + str(lambda3) + "   " + str(
        lambda4) + "  n_fft的值为： " + str(n_fft) + " hop_length的值为： " + str(hop_length) + "\n" + "超参数C=" + str(
        C) + "时，" + name + "的{}模型的正确率: %.3f".format(i) % score
    with open(name + '_svm_result.txt', 'a+') as f:  # 设置文件对象
        f.write(str1 + '\n')  # 将字符串写文件中
        f.close()


import numpy as np
import scipy.io as io


def npToMat(path_data, path_label, name, i):
    data = np.load(path_data)
    labels = np.load(path_label)
    labels = labels + 1
    if name == 'Epilepsy':
        # 读取特征
        data = data.reshape(-1, 128 * 24)
        # 11392 - 2300 = 9092
        data_train = data[0:9092]
        data_test = data[9092:]
        labels = labels.ravel()
        data_label_train = labels[0:9092]
        data_label_test = labels[9092:]
    elif name == 'HAR':
        data = data.reshape(-1, 128 * 18)
        # 11392 - 2300 = 9092
        data_train = data[0:7255]
        data_test = data[7255:]
        labels = labels.ravel()
        data_label_train = labels[0:7255]
        data_label_test = labels[7255:]
    elif name == 'sleepEDF':
        data = data.reshape(-1, 128 * 127)
        # 11392 - 2300 = 9092
        data_train = data[0:33330]
        data_test = data[33330:]
        labels = labels.ravel()
        data_label_train = labels[0:33330]
        data_label_test = labels[33330:]
    elif name == 'origin_dataset2b':
        data = data.reshape(-1, 128 * 127)
        # 720里   都是704 batch_size=64, 32
        data_train = data[0:384]
        data_test = data[384:]
        labels = labels.ravel()
        data_label_train = labels[0:384]
        data_label_test = labels[384:]
    data_train = data_train.T
    data_test = data_test.T
    io.savemat('./mat/{}_{}.mat'.format(name, i), {'trainlabels': data_label_train, 'testlabels': data_label_test,
                                                   'NewTrain_DAT': data_train, 'NewTest_DAT': data_test})
    # x = io.loadmat('./mat/{}.mat'.format(name))
