from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
import joblib
import csv

training_dir = "udp_data.csv"        # 训练集地址
testing_dir = "test.csv"          # 测试集地址

traindata = pd.read_csv(training_dir)
testdata = pd.read_csv(testing_dir)

datas = traindata

# print(datas)
# datas.drop(['Unnamed: 0'], inplace=True, axis=1)
# print(datas)
x = datas.iloc[:, datas.columns != "label"]
# print(x)
y = datas.iloc[:, datas.columns == "label"]
# print(y)

x = preprocessing.scale(x)
# print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = DecisionTreeClassifier(max_depth=3, max_features=2)
# clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score = clf.score(x_test, y_test)
# print(score)
scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
# print(scores)

sum = 0
j = 0
for i in scores:
    sum+=i
    j+=1
result = sum/j
print(result)

k_range = range(1, 10)
k_score = []
k_loss = []
for k in k_range:
    knn = DecisionTreeClassifier(min_samples_leaf=k)
    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')     # for clf
    # loss = -cross_val_score(knn, x, y, cv=10, scoring='neg_mean_squared_error')    # regression
    k_score.append(scores.mean())
    # k_loss.append(loss.mean())

plt.plot(k_range, k_score)
plt.xlabel('value of k for DT')
plt.ylabel('Cross-validated Accuracy')
plt.show()

joblib.dump(clf, './model_save/DecisionTree.pkl')
#   restore
s = []
s.append("DecisionTree")
s.append(score)
s.append(result)

with open('./result/result.csv', 'a', newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(s)


