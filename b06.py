# 2折交叉验证（自重）
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
print('蒲程 20190565')
print('MPG与汽车自重的关系')
data = pd.read_csv('MPG.csv')
data = np.array(data)

x = data[:, 4]
y = data[:, 0]

kf = KFold(n_splits=2, random_state=None)  # 2折

i = 1
err_test_liner = np.zeros(2)
err_test_svr = np.zeros(2)
for train_index, test_index in kf.split(x, y):
    # print("Train:", train_index, "Validation:", test_index)
    print('     {} of kfold {}'.format(i, kf.n_splits))
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    x = x.reshape(-1, 1)

    model_lm = LinearRegression()
    model_lm.fit(x_train, y_train)
    Y_test_liner = model_lm.predict(x_test)

    model_svr = svm.SVR(C=15, epsilon=1)
    model_svr.fit(x_train, y_train)
    Y_test_svr = model_svr.predict(x_test)

    err_test_liner[i - 1] = mean_squared_error(y_test, Y_test_liner)
    err_test_svr[i - 1] = mean_squared_error(y_test, Y_test_svr)

    print('     线性回归模型的测试误差：%.5f' % (err_test_liner[i - 1]))
    print('     SVR模型的测试误差：%.5f' % (err_test_svr[i - 1]))
    i = i + 1

err_test_liner = np.mean(err_test_liner)
err_test_svr = np.mean(err_test_svr)
print('\n     线性回归模型的平均测试误差：%.5f' % (err_test_liner))
print('     SVR模型的平均测试误差：%.5f' % (err_test_svr))
