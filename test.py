# 作者: Peach Yang
# 2023年04月08日10时20分33秒
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


def finalAcc(path_features, path_labels, name):
    start_time = datetime.now()
    print('=============================================')
    print('正在读取SVM需要的特征：')
    data = np.load(path_features)
    # 读取数据标签
    labels = np.load(path_labels)
    if name == 'Epilepsy':
        # 读取特征
        data = data.reshape(-1, 128 * 35)
        # 11392 - 2300 = 9092
        data_train = data[0:9092]
        data_test = data[9092:]
        labels = labels.ravel()
        data_label_train = labels[0:9092]
        data_label_test = labels[9092:]
    elif name == 'HAR':
        data = data.reshape(-1, 128 * 27)
        # 11392 - 2300 = 9092
        data_train = data[0:7255]
        data_test = data[7255:]
        labels = labels.ravel()
        data_label_train = labels[0:7255]
        data_label_test = labels[7255:]
        HAR_label = np.load(path_labels)
    elif name == 'sleepEDF':
        data = data.reshape(-1, 128 * 127)
        # 11392 - 2300 = 9092
        data_train = data[0:33330]
        data_test = data[33330:]
        labels = labels.ravel()
        data_label_train = labels[0:33330]
        data_label_test = labels[33330:]
    elif name == 'pFD':
        data = data.reshape(-1, 128 * 162)
        # 11392 - 2300 = 9092
        data_train = data[0:10912]
        data_test = data[10912:]
        labels = labels.ravel()
        data_label_train = labels[0:10912]
        data_label_test = labels[10912:]
    i = 10.0
    print('=============================================')
    print('开始SVM训练：')
    print('=============================================')
    SVM(data_train, data_test, data_label_train, data_label_test, i, name)
    end_time = datetime.now() - start_time
    with open(name + '_svm_result.txt', 'a+') as f:
        f.write("运行总时间：" + '\n' + str(end_time) + '\n')
        f.close()


def SVM(epilepsy_data_train, epilepsy_data_test, epilepsy_data_label_train, epilepsy_data_label_test, C, name,):
    # 创建SVC/Support Vector Classification/支持向量机分类器模型
    svc_model = svm.SVC(C=C)
    # 将数据拟合到SVC模型中，此处用到了标签值HAR_label_train，是有监督学习
    svc_model.fit(epilepsy_data_train, epilepsy_data_label_train)
    score = svc_model.score(epilepsy_data_test, epilepsy_data_label_test)
    y_predict = svc_model.predict(epilepsy_data_test)
    print("超参数C=" + str(C) + "时，" + name + "的模型的正确率: %.3f" % score)
    str1 = "超参数C=" + str(C) + "时，" + name + "的模型的正确率: %.3f" % score
    with open(name + '_svm_result.txt', 'a+') as f:  # 设置文件对象
        f.write(str1 + '\n')  # 将字符串写文件中
        f.close()

# # [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# c = np.logspace(-2, 3, 6)
# for i in c:


# # KNN
# def KNN(n_neighbors):
#     knn = KNeighborsClassifier(n_neighbors=n_neighbors)  # 超参数N
#     # 利用训练数据拟合模型
#     knn.fit(HRA_train, HAR_label_train)
#     print("超参数n=" + str(n_neighbors) + "时，模型错误率：" + str(1 - knn.score(HAR_test, HAR_label_test)))
#     return 1 - knn.score(HAR_test, HAR_label_test)
# sleepEDF SVM，直接计算准确率 0.432

# correct = []
# for i in range(1, 11):
#     correct.append(KNN(i))
# estimator = KNeighborsClassifier()
# param_dict = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
# estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
# estimator.fit(HRA_train, HAR_label_train)
# y_predict = estimator.predict(HAR_test)
# score = estimator.score(HAR_test, HAR_label_test)
# print("knn 癫痫正确率： %.3f " % score)
