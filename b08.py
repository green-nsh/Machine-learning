import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('wine.txt')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
# print(data.head())
x = data[
    ['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'phenols', 'Flavanoids', 'Nonflavanoid', 'Proanthocyanins',
     'Color', 'Hue', 'diluted', 'Proline']]
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)

K = np.arange(2, 10)
err_train = []
err_test = []
best_k = [0, 0]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
i = 0
s = ''
for j in ['gini', 'entropy']:
    for k in K:
        model = tree.DecisionTreeClassifier(criterion=j, max_depth=k, random_state=1)
        model.fit(x_train, y_train)
        err_train.append(1 - model.score(x_train, y_train))
        err_test.append(1 - model.score(x_test, y_test))

    axes[i].grid(True, linestyle='-.')
    axes[i].plot(np.arange(2, 10), err_train, label='训练误差', color='deeppink', marker='o', linewidth=3, linestyle='-')
    axes[i].plot(np.arange(2, 10), err_test, label='测试误差', color='orchid', marker='o', linewidth=3, linestyle='-.')
    if i == 0:
        s = 'CART'
    else:
        s = 'ID3'
    axes[i].set_xlabel("决策树深度")
    axes[i].set_ylabel("误差")
    axes[i].set_title("{0}决策树深度与误差".format(s))
    axes[i].legend()
    best_k[i] = K[err_test.index(np.min(err_test))]
    err_train = []
    err_test = []
    i = i + 1

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.show()
figure1=fig.get_figure()
figure1.savefig('a333.png',dpi=800)

print("20190565蒲程")
print("CART决策树最优深度：%d" % (best_k[0]))
print("ID3决策树最优深度：%d" % (best_k[1]))
print('\n')

i = 0
err_train_ = []
err_test_ = []
for v in ['gini', 'entropy']:
    model_ = tree.DecisionTreeClassifier(criterion=v, max_depth=best_k[i], random_state=1)
    model_.fit(x_train, y_train)
    err_train_.append(1 - model_.score(x_train, y_train))
    err_test_.append(1 - model_.score(x_test, y_test))

    labels = ['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'phenols', 'Flavanoids', 'Nonflavanoid',
              'Proanthocyanins',
              'Color', 'Hue', 'diluted', 'Proline']
    classes = ['1', '2', '3']
    data_ = tree.export_graphviz(model_, feature_names=labels, class_names=classes, filled=True, rounded=True,
                                 out_file=None)
    graph = graphviz.Source(data_)
    if i == 0:
        tree_name = 'CART_DecisionTree'
    else:
        tree_name = 'ID3_DecisionTree'
    print("20190565蒲程")
    graph.render(tree_name)
    print(tree_name + ':')
    print("最优深度：%d" % (best_k[i]))
    print("训练误差：%.4f" % (err_train_[i]))
    print("测试误差：%.4f" % (err_test_[i]))
    print('\n' + tree.export_text(model_))
    i = i + 1
