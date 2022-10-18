from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import csv
from sklearn.model_selection import GridSearchCV


training_dir = "voice_data.csv"        # 训练集地址

traindata = pd.read_csv(training_dir)

datas = traindata
# print(datas)
# datas.drop(['Unnamed: 0'], inplace=True, axis=1)

x = datas.iloc[:, datas.columns != "label"]
# print(x)
y = datas.iloc[:, datas.columns == "label"]
# print(y)

x = preprocessing.scale(x)
# print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
knn = KNeighborsClassifier(n_neighbors=16)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
# print(y_pred)
score = knn.score(x_test, y_test)
print(score)

scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy').mean()
print(scores)

grid_param = {'n_neighbors': list(range(2, 11)),
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid = GridSearchCV(knn, grid_param, cv=5)
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
print(grid.cv_results_['params'])
print(grid.cv_results_['mean_test_score'])
k_range = range(1, 31)
k_score = []
k_loss = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')     # for clf
    loss = -cross_val_score(knn, x, y, cv=10, scoring='neg_mean_squared_error')    # regression
    k_score.append(scores.mean())
    k_loss.append(loss.mean())

plt.plot(k_range, k_score)
plt.xlabel('value of k for KNN')
plt.ylabel('Cross-validated Accuracy')
plt.show()

joblib.dump(knn, './model_save/KNN.pkl')
#   restore

s = []
s.append("KNN")
s.append(score)
s.append(scores)
# print(s)

with open('./result/result.csv', 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['algorithm', 'Train Accuracy', 'Test Accuracy'])
    writer.writerow(s)

