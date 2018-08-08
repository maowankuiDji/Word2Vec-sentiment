import numpy as np
import pandas as pd
import jieba
from gensim.models import word2vec
from sklearn.model_selection import train_test_split

#读入数据

pos = pd.read_table('E:/NLP/chinese-w2v-sentiment/data/pos.csv',header=None,index_col=None)
neg = pd.read_table('E:/NLP/chinese-w2v-sentiment/data/neg.csv',header=None,index_col=None)
print(pos.head())
#分词
pos['c_w'] = [jieba.lcut(sent) for sent in pos[0]]
neg['c_w'] = [jieba.lcut(sent) for sent in neg[0]]
#合并neg和pos
pos_and_neg = np.append(pos['c_w'],neg['c_w'],axis=0)
#pos_and_neg = np.concatenate((pos['cut_word'],neg['cut_word']))
#pos_and_neg
#构造对应的标签数组
table = np.append((np.ones(len(pos))),(np.zeros(len(neg))),axis=0)
#table.shape
#切分训练集合测试集
x_train,x_test,y_train,y_test = train_test_split(pos_and_neg,table,test_size=0.2)
#保存数据
def save_data(doc_path,data):
    np.save(doc_path,data)

x_train_data = save_data('E:/NLP/chinese-w2v-sentiment/data/x_train_data.npy',x_train)
x_test_data = save_data('E:/NLP/chinese-w2v-sentiment/data/x_test_data.npy',x_test)
y_train_data = save_data('E:/NLP/chinese-w2v-sentiment/data/y_train_data.npy',y_train)
y_test_data = save_data('E:/NLP/chinese-w2v-sentiment/data/y_test_data.npy',y_test)