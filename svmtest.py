# https://qiita.com/tomov3/items/039d4271ed30490edf7b
# coding: UTF-8

##### training section #####
# 必要なライブラリの import
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# データのロード
iris = load_iris()

# データの分割
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

# training set を用いて学習
logreg = LogisticRegression().fit(X_train, y_train)

# test set を用いて評価
score = logreg.score(X_test, y_test)
print('Test set score: {}'.format(score))

##### validation section #####
from sklearn.model_selection import cross_val_score
logreg = LogisticRegression()
# 交差検証
scores = cross_val_score(logreg, iris.data, iris.target)
# 各分割におけるスコア
print('Cross-Validation scores: {}'.format(scores))
# スコアの平均値
import numpy as np
print('Average score: {}'.format(np.mean(scores)))

##### grid search #####
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# パラメータを dict 型で指定
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],  'gamma' : [0.001, 0.01, 0.1, 1, 10, 100]}

# validation set は GridSearchCV が自動で作成してくれるため，
# training set と test set の分割のみを実行すればよい
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# fit 関数を呼ぶことで交差検証とグリッドサーチがどちらも実行される
grid_search.fit(X_train, y_train)

print('Test set score: {}'.format(grid_search.score(X_test, y_test)))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Best cross-validation: {}'.format(grid_search.best_score_))