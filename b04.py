# 机器学习第一次实验（马力）不同训练样本的对应图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import mean_squared_error

print('蒲程 20190565')
print('MPG与汽车马力的关系')

data = pd.read_csv('MPG.csv')
data = np.array(data)
data = pd.DataFrame(data)
data.dropna(inplace=True)
data = np.array(data)

x = data[:, 3]
y = data[:, 0]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
for i, H, L in [(0.10, 0, 0), (0.4, 0, 1), (0.95, 1, 0), (0.7, 1, 1)]:
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=i, random_state=0)
    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    x = x.reshape(-1, 1)

    model_lm = LinearRegression()
    model_lm.fit(x_train, y_train)

    x=sorted(x)

    model_svr = svm.SVR(C=1, epsilon=0.1)
    model_svr.fit(x_train, y_train)
    axes[H, L].scatter(x_train, y_train, color='deeppink', s=20, label='训练样本')
    axes[H, L].scatter(x_test, y_test, color='darkslateblue', s=20, marker='*', label='测试样本')
    axes[H, L].scatter(np.array(x)[model_svr.support_], y[model_svr.support_], marker='p', s=120, alpha=0.2, label='支持向量')
    axes[H, L].plot(np.array(x), model_svr.predict(np.array(x)), linestyle='-', label='SVR')
    axes[H, L].plot(np.array(x), model_lm.predict(np.array(x)), linestyle='-', label='线性回归', linewidth=1)
    axes[H, L].legend()

    Y_train = model_svr.predict(x_train)
    Y_test = model_svr.predict(x_test)

    err_train = mean_squared_error(y_train, Y_train)
    err_test = mean_squared_error(y_test, Y_test)
    axes[H, L].set_title("训练样本占比=%.2f,训练误差=%.2f,测试误差=%.2f" % (i, err_train, err_test))
    axes[H, L].set_xlabel('汽车马力')
    axes[H, L].set_ylabel('MPG')
    axes[H, L].grid(True, linestyle='-.')
    x = data[:, 3]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.show()
figure1=fig.get_figure()
figure1.savefig('a444.png',dpi=400)

# 选取一组数据点进行预测，输出均值和方差
xx = np.random.choice(x, 5)
xx = xx.reshape(-1, 1)
yy_svr = model_svr.predict(xx)
yy_line = model_lm.predict(xx)

print("    线性回归的均值为：%.3f, 方差为：%.3f" % (np.mean(yy_line), np.var(yy_line)))
print("    支持向量回归的均值为：%.3f, 方差为：%.3f" % (np.mean(yy_svr), np.var(yy_svr)))