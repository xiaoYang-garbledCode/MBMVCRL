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
from sklearn.metrics import f1_score
from sklearn import svm
from datetime import datetime
import scipy.io as io


def finalAcc(path_features, path_labels, name):
    start_time = datetime.now()
    print('=============================================')
    print('正在读取SVM需要的特征：')
    data = np.load(path_features)
    # 读取数据标签
    labels = np.load(path_labels)
    if name == 'Epilepsy':
        # 读取特征
        data = data.reshape(-1, 128 * 150)
        # 11392 - 2300 = 9092 bs=128
        # 11488 - 2300 = 9188 bs = 32
        data_train = data[0:9092]
        data_test = data[9092:]
        labels = labels.ravel()
        data_label_train = labels[0:9092]
        data_label_test = labels[9092:]
    elif name == 'HAR':
        data = data.reshape(-1, 128 * 144)
        # 10240 - 7293 = 2974  bs = 128
        data_train = data[0:]
        data_test = torch.load('./trainer/test.pt')['samples']
        data_test = data_test.reshape(-1, 9 * 128)
        data_label_test = torch.load('./trainer/test.pt')['labels']
        data_test = data_test[0:]
        labels = labels.ravel()
        data_label_train = labels[0:]
        data_label_test = data_label_test[0:]
    elif name == 'sleepEDF':
        data = data.reshape(-1, 128 * 114)
        #  # 42304 - 8910 = 33394
        #  42304 - 8000 = 34304
        # 42304
        # 42304 - 8000 = 34304
        # data_train = data[0:8000]
        # data_test = data[8000:]
        # labels = labels.ravel()
        # data_label_train = labels[0:8000]
        # data_label_test = labels[8000:]
        data_train = data[0:33394]
        data_test = data[33394:]
        labels = labels.ravel()
        data_label_train = labels[0:33394]
        data_label_test = labels[33394:]
    elif name == 'pFD':
        data = data.reshape(-1, 128 * 162)
        # 11392 - 2300 = 9092
        data_train = data[0:10912]
        data_test = data[10912:]
        labels = labels.ravel()
        data_label_train = labels[0:10912]
        data_label_test = labels[10912:]
    elif name == 'origin_dataset2b':
        data = data.reshape(-1, 128 * 127)
        # 720里   都是704 batch_size=64, 32
        data_train = data[0:384]
        data_test = data[384:]
        labels = labels.ravel()
        data_label_train = labels[0:384]
        data_label_test = labels[384:]
    elif name == 'origin_dataset2a':
        data = data.reshape(-1, 128 * 127)
        # 720里   都是704 batch_size=64, 32
        data_train = data[0:288]
        data_test = data[288:]
        labels = labels.ravel()
        data_label_train = labels[0:288]
        data_label_test = labels[288:]
    s = 1
    print('=============================================')
    print('开始SVM训练：')
    print('=============================================')
    SVM(data_train, data_test, data_label_train, data_label_test, s, name)
    end_time = datetime.now() - start_time
    with open(name + '_svm_result.txt', 'a+') as f:
        f.write("运行总时间：" + '\n' + str(end_time) + '\n')
        f.close()


def SVM(epilepsy_data_train, epilepsy_data_test, epilepsy_data_label_train, epilepsy_data_label_test, C, name):
    # 创建SVC/Support Vector Classification/支持向量机分类器模型
    svc_model = svm.SVC(C=C)
    # 将数据拟合到SVC模型中，此处用到了标签值HAR_label_train，是有监督学习
    svc_model.fit(epilepsy_data_train, epilepsy_data_label_train)
    score = svc_model.score(epilepsy_data_test, epilepsy_data_label_test)
    y_predict = svc_model.predict(epilepsy_data_test)
    f1_macro = f1_score(epilepsy_data_label_test, y_predict, average='macro')
    print("超参数C=" + str(C) + "时，" + name + "的模型的正确率: %.5f" % score)
    print("超参数C=" + str(C) + "时，" + name + "的模型MF1的值为: %.5f" % f1_macro)
    str1 = "超参数C=" + str(C) + "时，" + name + "的模型的正确率: %.5f" % score
    str3 = "超参数C=" + str(C) + "时，" + name + "的模型MF1的值为: %.5f" % f1_macro
    str2 = str(score)
    with open(name + '_svm_result.txt', 'a+') as f:  # 设置文件对象
        f.write(str1 + '\n' + str3 + '\n')  # 将字符串写文件中
        f.close()
    with open(name + 'summary_acc.txt', 'a+') as f:  # 设置文件对象
        f.write(str2 + '\n')  # 将字符串写文件中
        f.close()


def npToMat(path_data, path_label, name):
    data = np.load(path_data)
    labels = np.load(path_label)
    labels = labels + 1
    if name == 'Epilepsy':
        # 读取特征
        data = data.reshape(-1, 128 * 150)
        # 11392 - 2300 = 9092 bs=128
        # 11488 - 2300 = 9188 bs = 32
        data_train = data[0:9092]
        data_test = data[9092:]
        labels = labels.ravel()
        data_label_train = labels[0:9092]
        data_label_test = labels[9092:]
    elif name == 'HAR':
        data = data.reshape(-1, 128 * 144)
        # 10240 - 7293 = 2974  bs = 128
        data_train = data[0:7293]
        data_test = data[7293:]
        labels = labels.ravel()
        data_label_train = labels[0:7293]
        data_label_test = labels[7293:]
    elif name == 'sleepEDF':
        data = data.reshape(-1, 128 * 114)
        # 11392 - 2300 = 9092
        data_train = data[0:33394]
        data_test = data[33394:]
        labels = labels.ravel()
        data_label_train = labels[0:33394]
        data_label_test = labels[33394:]
    elif name == 'origin_dataset2b':
        data = data.reshape(-1, 128 * 132)
        # 720里   都是704 batch_size=64, 32
        data_train = data[0:384]
        data_test = data[384:]
        labels = labels.ravel()
        data_label_train = labels[0:384]
        data_label_test = labels[384:]
    elif name == 'origin_dataset2a':
        data = data.reshape(-1, 128 * 127)
        # 720里   都是704 batch_size=64, 32
        data_train = data[0:288]
        data_test = data[288:]
        labels = labels.ravel()
        data_label_train = labels[0:288]
        data_label_test = labels[288:]
    data_train = data_train.T
    data_test = data_test.T
    io.savemat('./mat/{}.mat'.format(name), {'trainlabels': data_label_train, 'testlabels': data_label_test,
                                             'NewTrain_DAT': data_train, 'NewTest_DAT': data_test})
