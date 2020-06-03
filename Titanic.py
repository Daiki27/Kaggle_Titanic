# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ＊列によっては：文字型になっているから, それを数値で置き換える.

import os
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

#データの読み込み.
currentDir = os.getcwd()
df = pd.read_csv(currentDir + "/DATA/train.csv")
df
#print(df.isnull().sum()) # Age:177, Cabin682, Embarked:2

train = df.copy()
train.drop(["Name", "Ticket", "Cabin"], axis = 1, inplace = True)
train["Age"].fillna(train["Age"].mean(), inplace=True)
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True) #最頻値[0]必要なんでや.
train["Sex"].replace("male", 0, inplace=True)
train["Sex"].replace("female", 1, inplace = True)
train["Embarked"].replace("Q", 0, inplace=True) #なぜEmbarkedの整数型が少数なの？
train["Embarked"].replace("C", 1, inplace=True)
train["Embarked"].replace("S", 2, inplace=True)
train
#print(train.isnull().sum())

col_explain = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train[col_explain]
y = train["Survived"]
model  = LogisticRegression(solver='lbfgs', max_iter = 500)
for k in range(len(col_explain)):
    selected_explanatory_var = []
    selector = SelectKBest(score_func=f_classif, k=k+1)
    selector.fit(X,y)
    mask = selector.get_support()
    a = np.where(mask==True)
    for i in range(len(a[0])):
        selected_explanatory_var.append(col_explain[a[0][i]])
    #ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
    # max_iter = 100 => max_iter = 500
    result = model.fit(train[selected_explanatory_var], y)
    print("k=", i+1, selected_explanatory_var)
    #print("係数",result.coef_, "切片",result.intercept_)
    print(model.score(train[selected_explanatory_var],y))

#k=7が決定係数が一番良い. k= 7 ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
model.fit(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], y)
df_test = pd.read_csv(currentDir + "/DATA/test.csv")
valid = df_test.copy()
#df_test.isnull().sum() =>Age86, Fare1, Cabin327
valid.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
valid["Age"].fillna(valid["Age"].mean(), inplace=True)
valid["Fare"].fillna(valid["Fare"].mean(), inplace=True)
valid["Sex"].replace("male",   0, inplace = True)
valid["Sex"].replace("female", 1, inplace = True)
valid["Embarked"].replace("Q", 0, inplace=True) #なぜEmbarkedの整数型が少数なの？
valid["Embarked"].replace("C", 1, inplace=True)
valid["Embarked"].replace("S", 2, inplace=True)
valid

predict = model.predict(valid)
submit_csv = pd.concat([df_test['PassengerId'], pd.Series(predict)], axis=1)
submit_csv.columns = ['PassengerId', 'Survived']
submit_csv.to_csv('./submition.csv', index=False)

#欠損値を持つ列名の取得.
rack_colmname = []
for i in range(len(df.columns)):
    colname = df.columns[i]
    if(df[colname].isnull().any()==True):
        rack_colmname.append(colname)
print(rack_colmname)

#ランダムフォレストのモデル.
RandomForest = RandomForestClassifier(
    bootstrap=True,      #ブートストラップサンプルをするかどうか.
    criterion='mse',     #control how the decision tree algorithm splits nodes:
    max_depth=None,      #決定木の深さの最大値.
    max_features='auto', #defalut,最適な分割をするために考慮する特徴量の数
    max_leaf_nodes=None, #default,生成される木の最大の葉の数.
    min_impurity_split=1e-07, #not, default:0, 
    min_samples_leaf=1,       #
    min_samples_split=2,      #サイトによって違う？このパラメータ値以上のデータ数を持たない分岐先は,それ以上の条件分岐がされず,葉となる.
    min_weight_fraction_leaf=0.0, #default, 外れ値??
    n_estimators=10,   #not, default = 100
    n_jobs=1,          #defalut, 並列処理するジョブの数, 
    oob_score=True,    #not, defalut:false, 汎化性能を検証するためにout-of-bag samplesを使うかどうか.
    random_state=None, #
    verbose=0,         #defalut, fittingとpredicttinのときの冗長性.
    warm_start=False   #defalut, ??
)



