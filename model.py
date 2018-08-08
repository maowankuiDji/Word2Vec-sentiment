#coding:utf-8

import numpy as np
import pandas as pd
from gensim.models import word2vec

#导入数据
def load_data(path):
    data = np.load(path)
    return data
x_train = load_data('E:/NLP/chinese-w2v-sentiment/data/x_train_data.npy')
x_test = load_data('E:/NLP/chinese-w2v-sentiment/data/x_test_data.npy')
y_train = load_data('E:/NLP/chinese-w2v-sentiment/data/y_train_data.npy')
y_test = load_data('E:/NLP/chinese-w2v-sentiment/data/y_test_data.npy')

#导入模型
train_model = word2vec.Word2Vec.load('E:/NLP/chinese-w2v-sentiment/train_model.model')
test_model = word2vec.Word2Vec.load('E:/NLP/chinese-w2v-sentiment/test_model.model')
#计算词向量
def get_sent_vec(size,sent,model):
    vec = np.zeros(size).reshape(1,size)
    count = 0
    for word in sent:
        try:
            vec += model[word].reshape(1,size)
            count += 1
        except:
            continue
    if count != 0:
        vec /= count
    return vec

def get_train_vec(x_train,x_test,train_model,test_model):
    train_vec = np.concatenate([get_sent_vec(300,sent,train_model) for sent in x_train])
    test_vec = np.concatenate([get_sent_vec(300,sent,test_model) for sent in x_test])
    #保存数据
    np.save('E:/NLP/chinese-w2v-sentiment/train_vec.npy',train_vec)
    np.save('E:/NLP/chinese-w2v-sentiment/test_vec.npy',test_vec)
    return train_vec,test_vec
###计算每句话的词向量
train_vec,test_vec = get_train_vec(x_train,x_test,train_model,test_model)
#print(x_train.shape,train_vec.shape)