# 机器学习第一次实验（自重）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm

import sklearn
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

data = pd.read_csv('MPG.csv')
data = np.array(data)

x = data[:, 4]
y = data[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
x = x.reshape(-1, 1)

model_lm = LinearRegression()
model_lm.fit(x_train, y_train)

x[:, 0].sort()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
for C, E, H, L in [(1, 0.1, 0, 0), (1, 100, 0, 1), (100, 0.1, 1, 0), (10000, 0.01, 1, 1)]:
    model_svr = svm.SVR(C=C, epsilon=E)
    model_svr.fit(x_train, y_train)
    axes[H, L].scatter(x_train, y_train, color='deeppink', s=20,label='训练样本')
    axes[H, L].scatter(x_test, y_test, color='darkslateblue', s=20, marker='*',label='测试样本')
    axes[H, L].scatter(x[model_svr.support_], y[model_svr.support_], marker='p', s=120, alpha=0.2,label='支持向量')
    axes[H, L].plot(x, model_svr.predict(x), linestyle='-', label='SVR', linewidth=3)
    axes[H, L].plot(x, model_lm.predict(x), linestyle='-', label='线性回归', linewidth=3)
    axes[H, L].legend()

    Y_train = model_svr.predict(x_train)
    Y_test = model_svr.predict(x_test)

    err_train = mean_squared_error(y_train, Y_train)
    err_test = mean_squared_error(y_test, Y_test)
    axes[H, L].set_title("C=%d,epsilon=%.2f,训练误差=%.2f,测试误差=%.2f" % (C, E, err_train, err_test))
    axes[H, L].set_xlabel('汽车自重')
    axes[H, L].set_ylabel('MPG')
    axes[H, L].grid(True, linestyle='-.')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.show()
figure1=fig.get_figure()
figure1.savefig('a11111.png',dpi=400)

