#coding:utf-8

import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.svm import SVC
from sklearn.externals import joblib

#下载数据
#def get_data():
train_vec = np.load('E:/NLP/chinese-w2v-sentiment/train_vec.npy')
test_vec = np.load('E:/NLP/chinese-w2v-sentiment/test_vec.npy')
y_train =np.load('E:/NLP/chinese-w2v-sentiment/data/y_train_data.npy')
y_test = np.load('E:/NLP/chinese-w2v-sentiment/data/y_test_data.npy')
#训练SVM模型
def svm_tran(train_vec,y_train,test_vec,y_test):
    clf = SVC(kernel='rbf',verbose=True)
    clf.fit(train_vec,y_train)
    #持久化保存模型
    joblib.dump(clf,'E:/NLP/chinese-w2v-sentiment/svm_model/svm_model.pkl',compress=3)
    print(clf.score(test_vec,y_test))

#svm_tran(train_vec,y_train,test_vec,y_test)