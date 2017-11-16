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

#%%----------------------------gridSearch--------------------------------------
## Reference code from http://scikit-learn.org/stable/modules/grid_search.html

import matplotlib.pyplot as plt
import sklearn as sk
from numpy import genfromtxt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import Normalize

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

X,y = tr_data, tr_label

# A logarithmic grid with basis 10 for coarse search
C_range = np.logspace(-3, 5, 8)
gamma_range = np.logspace(-5, 3, 8)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best classifier is: ", grid.best_estimator_)

# Extract scores
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()
#%%---------------------------Ensemble-----------------------------------------
# Random Forest Classifier
n_clf = 6
tr_accuracy = np.zeros((n_clf, n))
val_accuracy = np.zeros((n_clf, n))
pd.options.display.float_format = '{:,.4f}'.format
result = pd.DataFrame(np.zeros((n_clf, 2)), 
                   index = range(n_clf), 
                   columns = ['Train Accuracy',
                              'Validation Accuracy'])

cv = 0
for tr, val in kf.split(tr_data, tr_label):
    tr_attrs = tr_data[tr, :]
    tr_target = tr_label[tr]
    val_attrs = tr_data[val, :]
    val_target = tr_label[val]
    
    # Tuning some parameters for random forest classifiers
    models = (RandomForestClassifier(n_estimators = 30, criterion = 'entropy', min_samples_split=10),
             RandomForestClassifier(n_estimators = 30, criterion = 'gini', min_samples_split=10),
             RandomForestClassifier(n_estimators = 50, criterion = 'entropy', min_samples_split=10),
             RandomForestClassifier(n_estimators = 100, criterion = 'entropy', min_samples_split=10),
             RandomForestClassifier(n_estimators = 100, criterion = 'entropy', min_samples_split=20),
             RandomForestClassifier(n_estimators = 100, criterion = 'entropy'))
    models = (clf.fit(tr_attrs, tr_target) for clf in models)
    # Calculate accuracy on training set and validation set
    for j, clf in zip(range(n_clf), models):
        tr_accuracy[j, cv] = np.mean(clf.predict(tr_attrs) == tr_target)
        val_accuracy[j, cv] = np.mean(clf.predict(val_attrs) == val_target)
    cv += 1
    
result['Train Accuracy'] = np.mean(tr_accuracy, axis=1)
result['Validation Accuracy'] = np.mean(val_accuracy, axis=1)
print(result)
# output
#   Train Accuracy  Validation Accuracy
#0          0.9877               0.9478
#1          0.9864               0.9466
#2          0.9888               0.9494
#3          0.9892               0.9519
#4          0.9803               0.9457
#5          0.9997               0.9509