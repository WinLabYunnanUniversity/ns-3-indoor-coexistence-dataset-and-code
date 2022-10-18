from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
data = testdata

# print(datas)
# datas.drop(['Unnamed: 0'], inplace=True, axis=1)
# print(datas)
x = datas.iloc[:, datas.columns != "label"]
print(x)
y = datas.iloc[:, datas.columns == "label"]
# print(y)

x = preprocessing.scale(x)
print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = RandomForestClassifier(n_estimators=20, max_depth=2, min_samples_leaf=2,
                             criterion="gini"
                             )
# clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(y_pred)
score = clf.score(x_test, y_test)
print(score)
scores = max(cross_val_score(clf, x, y, cv=5, scoring='accuracy'))
print(scores)

# k_range = range(1, 100)
# k_score = []
# k_loss = []
# for k in k_range:
#     knn = RandomForestClassifier(n_estimators=k)
#     scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')     # for clf
#     # loss = -cross_val_score(knn, x, y, cv=10, scoring='neg_mean_squared_error')    # regression
#     k_score.append(scores.mean())
#     # k_loss.append(loss.mean())
#
# plt.plot(k_range, k_score)
# plt.xlabel('value of k for RT')
# plt.ylabel('Cross-validated Accuracy')
# plt.show()


joblib.dump(clf, './model_save/RadomForest_udp.pkl')
#   restore

# s = []
# s.append("RandomForest")
# s.append(scores)
# s.append(score)
# # print(s)
#
# with open('./result/result.csv', 'a', newline="") as csvfile:
#     writer = csv.writer(csvfile)

