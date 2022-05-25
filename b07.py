# 作业第二题
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel('ERROR')

data = pd.read_csv('wine.txt')
x = data.iloc[:, 1:3].values
y = data.iloc[:, 0].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
i = 0
err_train = np.array(4)
err_test = 0
print("19机器人系1班—20190565—蒲程")
for v, H, L in [('linear', 0, 0), ('poly', 0, 1), ('sigmoid', 1, 0), ('rbf', 1, 1)]:
    classifier = SVC(kernel=v, random_state=0)
    classifier.fit(x_train, y_train)
    err_train = 1 - classifier.score(x_train, y_train)
    err_test = 1 - classifier.score(x_test, y_test)

    print("核函数为：" + v)
    print("    训练误差：%.4f" % (err_train))
    print("    测试误差：%.4f" % (err_test))

    x_set_train, y_set_train = x_train, y_train
    x_set_test, y_set_test = x_test, y_test

    X1, X2 = np.meshgrid(np.arange(start=x_set_train[:, 0].min() - 1, stop=x_set_train[:, 0].max() + 1, step=0.01),
                         np.arange(start=x_set_train[:, 1].min() - 1, stop=x_set_train[:, 1].max() + 1, step=0.01))
    axes[H, L].contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                        alpha=0.75, cmap=ListedColormap(('fuchsia', 'mediumspringgreen', 'deepskyblue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set_train)):
        axes[H, L].scatter(x_set_train[y_set_train == j, 0], x_set_train[y_set_train == j, 1],
                           c=ListedColormap(('crimson', 'green', 'dodgerblue'))(i), label=str(j) + '（训练样本）')
    for i, j in enumerate(np.unique(y_set_test)):
        axes[H, L].scatter(x_set_test[y_set_test == j, 0], x_set_test[y_set_test == j, 1],
                           c=ListedColormap(('crimson', 'green', 'dodgerblue'))(i), marker='*',
                           label=str(j) + '（测试样本）', s=30)

    axes[H, L].set_title("SVM:kernel=" + v)
    axes[H, L].set_xlabel("Alcohol")
    axes[H, L].set_xlabel("Malic acid")
    axes[H, L].legend()
    i + 1
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.show()
figure1=fig.get_figure()
figure1.savefig('a7.png',dpi=800)
