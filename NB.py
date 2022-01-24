from nltk.corpus import stopwords
import file
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import utils
import matplotlib.pyplot as plt
stops = set(stopwords.words('english'))

# 数据加载
train_df = file.load_file('./dataset/train_en.json')
test_df = file.load_file('./dataset/test_en.json')

train_data = []
test_data = []
train_label = []

for i in range(len(train_df)):
    train_data.append(train_df[i]['content'])
    train_label.append(train_df[i]['label'])
    train_data[i] = utils.gettext(train_data[i])
    train_data[i] = utils.text_process(train_data[i])
    train_data[i] = utils.getwords(train_data[i])
    train_data[i] = ' '.join(train_data[i])



for i in range(len(test_df)):
    test_data.append(test_df[i]['content'])
    test_data[i] = utils.gettext(test_data[i])
    test_data[i] = utils.text_process(test_data[i])
    test_data[i] = utils.getwords(test_data[i])
    test_data[i] = ' '.join(test_data[i])






# TF-IDF提取特征
tfidf_vector = TfidfVectorizer(stop_words=stops, max_features=7000,lowercase=False, sublinear_tf=True, max_df=0.8)
#划分训练集与测试集
#train_X,test_X, train_y, test_y = train_test_split(train_data,train_label,test_size=0.2,random_state=0)

tfidf_vector.fit(train_data)
train_tfidf = tfidf_vector.transform(train_data)
test_tfidf = tfidf_vector.transform(test_data)
clf_nb = MultinomialNB(alpha=0.04)  # 模型参数可以根据分类结果进行调优
clf_nb.fit(train_tfidf, train_label)  # 模型训练
y_pred = clf_nb.predict(test_tfidf)  # 模型预测
"""
time = []
accuracy = []
microf1 = []
macrof1 = []

for i in range(100):
    j=i/100
    tfidf_vector.fit(train_X)
    train_tfidf = tfidf_vector.transform(train_X)
    test_tfidf = tfidf_vector.transform(test_X)
    clf_nb = MultinomialNB(alpha=j)  # 模型参数可以根据分类结果进行调优
    clf_nb.fit(train_tfidf, train_y)  # 模型训练
    y_pred = clf_nb.predict(test_tfidf)  # 模型预测
    time.append(j)
    accuracy.append(accuracy_score(y_pred,test_y))
    microf1.append(f1_score(test_y,y_pred,average='micro'))
    macrof1.append(f1_score(test_y,y_pred,average='macro'))
    print('alpha =',j )
    print(accuracy_score(y_pred,test_y))
    

plt.plot(time , accuracy , label = 'accuracy',color = 'r')
plt.plot(time, microf1, label = 'microf1', color = 'g')
#plt.plot(time, macrof1, label = 'macrof1', color = 'b')
plt.xlabel('alpha')
plt.legend()
plt.show()"""
np.savetxt(fname="result.csv", X=y_pred, fmt="%d",delimiter="\n")

# 查看各类指标
#print(classification_report(y_test, y_pred,target_names=class_list,digits=4))

