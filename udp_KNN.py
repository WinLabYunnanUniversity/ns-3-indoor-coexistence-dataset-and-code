from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import csv

training_dir = "udp_data.csv"        # 训练集地址

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
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
score = knn.score(x_test, y_test)

scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
# print(scores)
sum = 0
j = 0
for i in scores:
    sum+=i
    j+=1
result = sum/j

k_range = range(1, 31)
k_score = []
k_loss = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')     # for clf
    loss = -cross_val_score(knn, x, y, cv=10, scoring='neg_mean_squared_error')    # regression
    k_score.append(scores.mean())
    k_loss.append(loss.mean())

plt.plot(k_range, k_loss)
plt.xlabel('value of k for KNN')
plt.ylabel('Cross-validated Accuracy')
plt.show()

joblib.dump(knn, './model_save/KNN.pkl')
#   restore

s = []
s.append("KNN")
s.append(result)
s.append(score)
# print(s)

with open('./result/result.csv', 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['algorithm', 'Train Accuracy', 'Test Accuracy'])
    writer.writerow(s)

