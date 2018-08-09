# Word2Vec-sentiment
## 基于Word2Vec+SVM对电商的评论数据进行情感分析
首先是利用word2vec对正负评论数据进行词向量训练，然后利用SVM分类器对语料进行分类，具体的过程如下：
### 第一步：加载数据、进行jieba分词、对数据进行随机切分，生成训练集和测试集（对应的代码部分为data_seal.py）
pos = pd.read_table('E:/NLP/chinese-w2v-sentiment/data/pos.csv',header=None,index_col=None)  
neg = pd.read_table('E:/NLP/chinese-w2v-sentiment/data/neg.csv',header=None,index_col=None)  
导入数据，然后利用jieba对数组进行分词，将分词结果与生成的相同维度的标签table数组进行合并，合并的方式有很多种：这里我用的是np.append(a,b,axis=0)
的方式。数据准备好了之后就是对数据进行切分，随机生成测试数据集和训练集，这里的比例test_size可以根据数据的实际大小进行设置，正常设置成0.2和0.3。  
为了后面的运算方便，对切分的数据进行保存。分别为data文件下的x_train_data、x_test_data、y_train_data、y_test_data。
### 第二步：计算每段话的向量（代码对应的是word_vec.py和model.py）
通过遍历每句话中每个词的词向量，然后求均值，将均值表示为这一句话对应的向量，当然这里只是简单初级的处理，也可以参考doc2vec的方法对
句子进行向量化，但是工程应用上的效果不是特别好，大家也可尝试一下看看，这里毕竟只是初级的教程。  
### 第三步：训练SVM模型（代码对应的是train_model.py）
### 第四步：对单个句子进行分类，情感判断（对应的代码为model_test.py）
