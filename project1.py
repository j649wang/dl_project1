# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:39:24 2017

@author: wangjingyi
"""

import numpy as np
import pandas as pd
import sklearn as sk

from numpy import genfromtxt
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Input
tr_data = genfromtxt('E:\\MSBDT\\6000B\\project1\\traindata.csv', delimiter=',')
tr_label = genfromtxt("E:\\MSBDT\\6000B\\project1\\trainlabel.csv", delimiter=',')
tst_data = genfromtxt("E:\\MSBDT\\6000B\\project1\\testdata.csv", delimiter=',')

# Normalization
all_data = np.append(tr_data, tst_data, axis = 0)
all_data = sk.preprocessing.scale(all_data)
tr_data = np.take(all_data, range(len(tr_data)), axis = 0)
tst_data = np.take(all_data, range(len(tst_data)), axis = 0)

# Classifiers
clf_name = pd.Series(['Naive Bayes',
                      'Logistic Regression',
                      'linear SVC',
                      'SVC with linear kernel',
                      'Initial SVC with RBF kernel',
                      'Tuned RBF SVC',
                      'Decision Tree with max depth 10',
                      'Decision Tree with max depth 15',
                      'Adaboost Decision tree with depth 5',
                      'Adaboost Decision tree with depth 10',
                      'Adaboost Decision tree with depth 15',
                      'Random Forest'])
n_clf = clf_name.size

# Conduct 10-Fold cross validation on training set
n = 10
kf = KFold(n_splits = n, shuffle = True)

# Evaluate each classifier's training accuracy and validation accuracy
tr_accuracy = np.zeros((n_clf, n))
val_accuracy = np.zeros((n_clf, n))
pd.options.display.float_format = '{:,.4f}'.format
result = pd.DataFrame(np.zeros((n_clf, 2)), 
                   index = clf_name, 
                   columns = ['Train Accuracy',
                              'Validation Accuracy'])

# Cross validation for multiple classifiers
cv = 0
for tr, val in kf.split(tr_data, tr_label):
    tr_attrs = tr_data[tr, :]
    tr_target = tr_label[tr]
    val_attrs = tr_data[val, :]
    val_target = tr_label[val]
    
    C = 1.0  # SVM regularization parameter
    models = (GaussianNB(),
              linear_model.LogisticRegression(),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='linear', C=C),
              svm.SVC(kernel='rbf', C=C),
              svm.SVC(C=100000.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False),
              DecisionTreeClassifier(max_depth=10),
              DecisionTreeClassifier(max_depth=15),
              AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm='SAMME'),
              AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), algorithm='SAMME'),
              AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), algorithm='SAMME'),
              RandomForestClassifier(n_estimators = 100, criterion = 'entropy', min_samples_split=10))
    models = (clf.fit(tr_attrs, tr_target) for clf in models)
    # Calculate accuracy on training set and validation set
    for j, clf in zip(range(n_clf), models):
        tr_accuracy[j, cv] = np.mean(clf.predict(tr_attrs) == tr_target)
        val_accuracy[j, cv] = np.mean(clf.predict(val_attrs) == val_target)
    cv += 1
    
result['Train Accuracy'] = np.mean(tr_accuracy, axis=1)
result['Validation Accuracy'] = np.mean(val_accuracy, axis=1)
print(result)

# Fit the best classifier using all training set data
best_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), algorithm='SAMME')
best_clf.fit(tr_data, tr_label)
# Accuracy on training set
np.mean(best_clf.predict(tr_data) == tr_label)
# Prediction on testing set
tst_pred = best_clf.predict(tst_data)
np.savetxt('E:\\MSBDT\\6000B\\project1\\project1_20469513.csv', tst_pred, delimiter = ',', fmt='%.d')
