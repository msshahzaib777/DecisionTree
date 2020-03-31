# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#https://archive.ics.uci.edu/ml/datasets/Planning+Relax
#https://archive.ics.uci.edu/ml/datasets/Adult
#https://archive.ics.uci.edu/ml/datasets/Avila
#https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra
#https://archive.ics.uci.edu/ml/datasets/Echocardiogram

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from matplotlib import pyplot as plt
import seaborn as sns

def HoldOut(X, y, pruning=0.015, iterate=100, kernel = "gini"):
    scores = []
    for i in range(iterate):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=i)
        clf = tree.DecisionTreeClassifier(criterion=kernel, ccp_alpha = pruning).fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    return scores

def FoldCV(X, y, pruning=0.015, CV=10, iterate=10, kernel = "gini"):
    scores = []
    for i in range(iterate):
        clf = tree.DecisionTreeClassifier(criterion=kernel, ccp_alpha = pruning)
        score_CV = cross_val_score(clf, X, y, cv=CV)
        scores.append(score_CV)
    return scores  

def Datasets_ML(dataset):
    scores = []
    
    X = dataset.iloc[:, :dataset.shape[1]-1]
    y = dataset.iloc[:, dataset.shape[1]-1]
    
    DS_HoldOut_ent = HoldOut(X, y, pruning = 0, kernel="entropy")
    scores.append(np.mean(DS_HoldOut_ent))
    DS_HoldOut_ent_pruning = HoldOut(X, y, kernel="entropy")
    scores.append(np.mean(DS_HoldOut_ent_pruning))
    
    DS_HoldOut_gini = HoldOut(X, y, pruning = 0)
    scores.append(np.mean(DS_HoldOut_gini))
    DS_HoldOut_gini_pruning = HoldOut(X, y)
    scores.append(np.mean(DS_HoldOut_gini_pruning))
    
    DS_CV_ent = FoldCV(X, y, pruning = 0, kernel="entropy")
    scores.append(np.mean(DS_CV_ent))
    DS_CV_ent_pruning = FoldCV(X, y, kernel="entropy")
    scores.append(np.mean(DS_CV_ent_pruning))
    
    DS_CV_gini = FoldCV(X, y, pruning = 0)
    scores.append(np.mean(DS_CV_gini))
    DS_CV_gini_pruning = FoldCV(X, y)
    scores.append(np.mean(DS_CV_gini_pruning))
    
    return scores

def outliers(dataset, to):
    for i in to:
        dataset.iloc[:, i].replace('?', np.nan, inplace=True)
        dataset.iloc[:, i] = dataset.iloc[:, i].astype(float)
        Q1 = dataset.iloc[:, i].quantile(0.25)
        Q3 = dataset.iloc[:, i].quantile(0.75)
        IQR = Q3 - Q1
        dataset.iloc[:, i].mask((dataset.iloc[:, i] < Q1-1.5*IQR) | (dataset.iloc[:, i] > Q3+1.5*IQR), inplace = True )
        dataset.iloc[:, i].interpolate(inplace = True)
    return dataset

dataset1 = pd.read_csv("plrx.txt", delimiter="\t", sep= " ", header = None)
dataset1 = dataset1.iloc[:, 0:13]
dataset1 = outliers(dataset1, range(0, 11))


dataset2 = pd.read_csv("dataset2/adult.data", delimiter=",", sep= " ", header = None)
dataset2.append(pd.read_csv("dataset2/adult.test", delimiter=",", sep= " ", header = None))
#
# Data set 2 Cleaning
#
value_counts1 = dataset2.iloc[:, 1].value_counts().index.tolist()
dataset2.iloc[:, 1].replace(value_counts1[:3], [0, 1, 2], inplace =True)
dataset2.iloc[:, 1].replace(value_counts1[3:], 3, inplace =True)

value_counts3 = dataset2.iloc[:, 3].value_counts().index.tolist()
dataset2.iloc[:, 3].replace(value_counts3, [0, 1, 2, 3, 4, 5, 6, 5, 5, 1, 5, 5, 3, 5, 5, 5], inplace =True)

value_counts5 = dataset2.iloc[:, 5].value_counts().index.tolist()
dataset2.iloc[:, 5].replace(value_counts5[:5], [0, 1, 2, 3, 4], inplace =True)
dataset2.iloc[:, 5].replace(value_counts5[5:], 5, inplace =True)

value_counts6 = dataset2.iloc[:, 6].value_counts().index.tolist()
dataset2.iloc[:, 6].replace(value_counts6, [0, 1, 2, 3, 4, 5, 6, 5, 7, 8, 5, 5, 5, 5, 5], inplace =True)

value_counts7 = dataset2.iloc[:, 7].value_counts().index.tolist()
dataset2.iloc[:, 7].replace(value_counts7, [0, 1, 2, 3, 4, 5], inplace =True)

value_counts8 = dataset2.iloc[:, 8].value_counts().index.tolist()
dataset2.iloc[:, 8].replace(value_counts8, [0, 1, 2, 3, 3], inplace =True)

dataset2.iloc[:, 9].replace([' Male', ' Female'], [0, 1], inplace =True)

value_counts13 = dataset2.iloc[:, 13].value_counts().index.tolist()
dataset2.iloc[:, 13].replace(value_counts13[:11], [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9], inplace =True)
dataset2.iloc[:, 13].replace(value_counts13[11:], 10, inplace =True)

value_counts14 = dataset2.iloc[:, 14].value_counts().index.tolist()
dataset2.iloc[:, 14].replace(value_counts14, [0, 1], inplace =True)

dataset3 = pd.read_csv("avila/avila-tr.txt", delimiter=",", header = None)
dataset3.append(pd.read_csv("avila/avila-ts.txt", delimiter=",", header = None))
dataset3 = outliers(dataset3, range(0, 9))
value_countDS3_10 = dataset3.iloc[:, 10].value_counts().index.tolist()
dataset3.iloc[:, 10].replace(value_countDS3_10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace =True)


dataset4 = pd.read_csv("dataR2.csv")
dataset4 = outliers(dataset4, range(0,8))


dataset5 = pd.read_csv("dataset5/echocardiogram.data", delimiter=",", sep=" ", header = None)
dataset5.drop(10, axis=1, inplace=True)
dataset5.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
dataset5 = outliers(dataset5, range(0, 10))
dataset5.iloc[:, 11].replace('?', np.nan, inplace=True)
dataset5.dropna(inplace= True)

s1 = Datasets_ML(dataset1)
s2 = Datasets_ML(dataset2)
s3 = Datasets_ML(dataset3)
s4 = Datasets_ML(dataset4)
s5 = Datasets_ML(dataset5)
    
df = pd.DataFrame([s1, s2, s3, s4, s5]).transpose()
    
df.columns = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4", "Dataset 5"]
df.index = ["HoldOut_Entropy", "HoldOut_Entropy_Pruning", "HoldOut_Gini", "HoldOut_Gini_Pruning", "FoldCV_Entropy", "FoldCV_Entropy_Pruning", "FoldCV_Gini", "FoldCV_Gini_Pruning"]    

print(df)

print("Std. of Dataset 1:", np.std(df.iloc[:, 0]))
print("Std. of Dataset 2:", np.std(df.iloc[:, 1]))
print("Std. of Dataset 3:", np.std(df.iloc[:, 2]))
print("Std. of Dataset 4:", np.std(df.iloc[:, 3]))
print("Std. of Dataset 5:", np.std(df.iloc[:, 4]))