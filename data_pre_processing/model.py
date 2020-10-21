import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVR

def train_model(model, hypers, X_train, y_train):
  if model == 'RF_class':
    criterion = hypers['criterion']
    n_estimators = hypers['n_estimators']
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    clf.fit(X_train, y_train)
    return clf

  elif model == 'KNN_class':
    n_neighbors = hypers['n_neighbors']
    weights = hypers['weights']
    algorithm = hypers['algorithm']
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    clf.fit(X_train, y_train)
    return clf

  elif model == 'SVM_class':
    penalty = hypers['penalty']
    loss = hypers['loss']
    clf = LinearSVC(penalty=penalty, loss=loss)
    clf.fit(X_train, y_train)
    return clf

  elif model == 'GB_class':
    loss = hypers['loss']
    learning_rate = hypers['learning_rate']
    n_estimators = hypers['n_estimators']
    clf = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    return clf

  elif model == 'RF_reg':
    criterion = hypers['criterion']
    n_estimators = hypers['n_estimators']
    reg = RandomForestRegressor(criterion=criterion, n_estimators=n_estimators)
    reg.fit(X_train, y_train)
    return reg
  
  elif model == 'SVM_reg':
    kernel = hypers['kernel']
    degree = hypers['degree']
    reg = SVR(kernel=kernel, degree=degree)
    reg.fit(X_train, y_train)
    return reg
  
  else:
    loss = hypers['loss']
    learning_rate = hypers['learning_rate']
    n_estimators = hypers['n_estimators']
    criterion = hypers['criterion']
    reg = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, criterion=criterion)
    reg.fit(X_train, y_train)
    return reg
  
def evaluate_model(model, X_test, y_test):
  score = model.score(X_test, y_test)
  return score
