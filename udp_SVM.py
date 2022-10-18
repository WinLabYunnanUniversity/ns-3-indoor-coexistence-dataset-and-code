from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
svm = SVC(gamma=0.1)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
score = svm.score(x_test, y_test)

scores = cross_val_score(svm, x, y, cv=5, scoring='accuracy')
sum = 0
j = 0
for i in scores:
    sum+=i
    j+=1
result = sum/j
print(result)
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color="r", label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g", label="Cross-validation")
plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()

joblib.dump(svm, './model_save/svm.pkl')
#   restore

s = []
s.append("svm")
s.append(score)
s.append(result)
# print(s)

with open('./result/result.csv', 'a', newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(s)
