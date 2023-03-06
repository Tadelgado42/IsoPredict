#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:54:57 2022

@author: thomasdelgado
"""

#############################
# Import required libraries #
#############################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from itertools import product
from sys import exit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


###################################
# Read data from a given csv file #
###################################

def readData(fileName):
  data = []
  with open(fileName, 'r', encoding='latin1') as f:
    for i,line in enumerate(f):
      l = line.strip().split(',')
      if i>0:
        l =  [float(x) if isFloat(x) else x for x in l]
        if any(l): data.append(l)
  return data

####################################################################
# Return true if x can be converted to a float and false otherwise #
####################################################################

def isFloat(x):
  try:
    float(x)
    return True
  except:
    return False

##################################################################
# Perfom train/test data split and save result as seperate files #
##################################################################

def splitData(fileName, fracTest=0.2):
  header = ''
  with open(fileName, 'r') as f: header = f.readline().strip()
  data = readData(fileName)
  train, test = train_test_split(data, test_size=fracTest)
  trainName = 'train_'+fileName
  with open(trainName, 'w') as o:
    o.write(header+'\n')
    for x in train: o.write(','.join(map(str, x))+'\n')
  testName = 'test_'+fileName
  with open(testName, 'w') as o:
    o.write(header+'\n')
    for x in test: o.write(','.join(map(str, x))+'\n')

#############################################################
# Select desired columns and remove rows with missing data  #
#############################################################

def filterData(data, indices, target=-1):

  #map regions to integers
  regions = {}
  if target!=-1:
    count = 0
    for row in data:
      if row[target] not in regions:
        regions[row[target]] = count
        count += 1
      row[target] = regions[row[target]]
  return regions, [[row[i] for i in indices] for row in data if all([isFloat(row[j]) for j in indices])]

#########################################
# Display confusion matrix as a heatmap #
#########################################

def confusionHeatmap(confusion, regions, figFile=''):

  (labels,_) = zip(*sorted([(k, regions[k]) for k in regions], key=lambda x: x[1]))
  M = [[np.mean(confusion[act][pred]) for pred in range(len(labels))] for act in range(len(labels))]
  M = pd.DataFrame(M, index=labels, columns=labels)
  plt.clf()
  sb.heatmap(M, annot=True, fmt='g')
  if figFile: plt.savefig(figFile, bbox_inches='tight')
  else: plt.show()
  
##########################################
# Test random classifier to ensure model # 
# accuracies are above random chance     # 
##########################################

def testRandomClassifier(train_y, reg_counts, k=5, weighted=False):

  size = int(len(train_y)/k)
  perm = np.random.permutation(train_y)
  pred = np.random.choice(len(reg_counts), p=[reg_counts[i]/len(train_y) for i in reg_counts], size=len(train_y)) if weighted else np.random.choice(len(reg_counts), size=len(train_y))
  act_folds = [perm[i*size:(i+1)*size] for i in range(k-1)] + [perm[(k-1)*size:]]
  pred_folds = [pred[i*size:(i+1)*size] for i in range(k-1)] + [pred[(k-1)*size:]]

  print(len(perm), [len(f) for f in act_folds])
  print('k-fold accs', [accuracy_score(act, pred) for (act, pred) in zip(act_folds, pred_folds)],
        '\ntotal acc', accuracy_score(perm, pred),
        '\nk-fold f1', [f1_score(act, pred, average='weighted') for (act, pred) in zip(act_folds, pred_folds)],
        '\ntotal f1', f1_score(perm, pred, average='weighted'))

###########################
# K-Fold Cross-Validation #
###########################

def runCV(X, Y, models, topN=1):
  for name in models:
    clf = GridSearchCV(estimator=models[name]['estimator'], param_grid=models[name]['param_grid'], scoring=make_scorer(accuracy_score), cv=5)
    clf.fit(X, Y)
    print('##########', name, '############')
    bestN = [list(clf.cv_results_['rank_test_score']).index(i) for i in range(1, topN+1, 1)]
    for i in bestN:
      print('Parameters:', clf.cv_results_['params'][i])
      print('Score:', clf.cv_results_['mean_test_score'][i])

if __name__=='__main__':

#####################################
#                                   #
#  Access, sort, and split the data #### If new datafile, uncomment splitData line 
#                                   #    If split already performed, comment out splitData
#####################################  

  fileName = 'INSERT FILE NAME HERE'

  #splitData(fileName, fracTest=0.2) 

  trainData = readData('train_'+fileName)

  indices = [1, 2, 3, 6, 7] # Specify columns for lat, long, region, H, O

  regions, trainData = filterData(trainData, indices, target=3)

  print('Number of rows in filtered data: ', len(trainData))

  print('Regions', regions)
  
  X = np.array([row[3:] for row in trainData], dtype='float64') # Use H and O as input

  Y = np.array([row[2] for row in trainData], dtype='float64')  # To try and predict region
  
  # testRandomClassifier(Y, regions, k=5, weighted=False)

  # testRandomClassifier(Y, reg_counts, k=5, weighted=True)
  
######################################
#                                    #
# Test for best ML hyperparameters   #                                 
#                                    #
######################################  

  models = {}

#######
# kNN #
#######
  
  # leaf_size = list(range(1,50))
  # n_neighbors = list(range(1,30))
  # p=[1,2]
  # models['knn'] = {'estimator':KNeighborsClassifier(), 
  #                  'param_grid':dict(n_neighbors=n_neighbors, 
  #                                    p=p)}
  
#######
# SVC #
#######

  # param_grid = {'C': [0.1, 1, 10, 100, 1000], 
  #              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
  #              'kernel': ['rbf']} 
  # models['svm'] = {'estimator':SVC(), 'param_grid':param_grid}

  
#######
# MLP #
#######

  # layers = [(2,), (4,), (8,)] + [(x,y) for x in [2,4,8] for y in [2,4,8]] + \
  #           [(x,y,z) for x in [2,4,8] for y in [2,4,8] for z in [2,4,8]] + \
  #           [(x,y) for x in [25, 50, 75, 100] for y in [10, 20, 30]]

  # param_grid = {'hidden_layer_sizes':layers,
  #              'activation':['relu', 'logistic', 'tanh'],
  #              'solver':['lbfgs', 'sgd'],
  #              'alpha':[0.001, 0.1, 1, 3, 5],
  #              'max_iter':[10000]}

  # models['ann'] = {'estimator':MLPClassifier(), 'param_grid':param_grid}

#######
# RFC #
#######

  # param_grid = {'n_estimators': [100, 200, 300],
  #               'criterion' : ['gini', 'entropy', 'log_loss'],
  #               'max_depth': [1,2,3,4,5,6,7,8,9,10],
  #               'min_samples_split': [8, 10, 12],
  #               'min_samples_leaf': [3, 4, 5],
  #               'max_features': ['sqrt', 'log2']} 
  
  # models['rfc'] = {'estimator':RandomForestClassifier(), 'param_grid':param_grid}

#######
# DTC #
#######

  # param_grid = {'criterion' : ['gini', 'entropy'],
                # 'max_depth': [1,2,3,4,5,6,7,8,9,10],
                # 'min_samples_split': [8, 10, 12],
                # 'min_samples_leaf': [3, 4, 5],
                # 'max_features': ['sqrt', 'log2']}

  # models['dtc'] = {'estimator':DecisionTreeClassifier(), 'param_grid':param_grid}

#######
# GNB #
#######
  
  # param_grid = { 'var_smoothing': np.logspace(0,-9, num=10)}

  # models['gnb'] = {'estimator':GaussianNB(), 'param_grid':param_grid}

##########
# Run CV #
##########
  # runCV(X, Y, models, topN=3)
  
###################################
#                                 #
#     Choose a ML algorithm       #### Test results from hyperparameter tuning      
#                                 #    on training data here   
###################################   

  clf = KNeighborsClassifier(leaf_size = 1, p = 1, n_neighbors=3)
  
  #clf = SVC(C = 1000, gamma=0.01, kernel='rbf', probability=True)
  
  #clf = MLPClassifier(hidden_layer_sizes=(50,30), activation = 'tanh', alpha =0.1,
  #                   solver='lbfgs', max_iter=10000)
  
  # clf = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='sqrt',
  #                          min_samples_split=8, min_samples_leaf=4, n_estimators=200) 
  
  #clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features='log2',
  #                             min_samples_split=10, min_samples_leaf=5)
  
  
  # clf = GaussianNB(var_smoothing=1.0)
  
##########################################
#                                        #
# Show k-folds accuracy on training data #    
#                                        #
########################################## 

  # print(len(X), len(Y))

  # accuracies = cross_val_score(clf, X, Y, scoring='accuracy', cv=5)

  # print('Accuracies:',accuracies, np.mean(accuracies))

  # clf.fit(X, Y)

  # y_hat = clf.predict(X)

  # M = confusion_matrix(Y, y_hat)

  # for row in M: print(row)

  # confusionHeatmap(M, regions, figFile='confusion.png')

  ###############################################################
  #                                                             #
  #  Apply to Test Set - this should be done only once after a  #
  #  final model is chosen                                      #
  #                                                             #
  ###############################################################

  # testData = readData('test_'+fileName)
  # testRegions, testData = filterData(testData, indices, target=3)
  # print('Regions', regions, testRegions) #make sure that the regions for test and train match
  
  # regionMap = {testRegions[r]:regions[r] for r in regions}
  # print('region map', regionMap)
  
  # clf.fit(X, Y)
  # testX = np.array([row[3:] for row in testData], dtype='float64') # Use H and O as input
  # testY = np.array([regionMap[row[2]] for row in testData], dtype='float64')  # To try and predict region
  # # testRandomClassifier(testY, regions, k=5, weighted=False)
  
  # y_hat = clf.predict(testX)
  # M = confusion_matrix(testY, y_hat)
  # for row in M: print(row)
  # confusionHeatmap(M, regions, figFile='test_confusion.png')

  # print('f1 score:', f1_score(testY, y_hat, average='weighted'))
  # print ('Accuracy:', accuracy_score(testY, y_hat))
  
  ###############################################################
  #                                                             #
  #  Apply to Cases - this should be done after testing and     #
  #  using all data                                             #
  #                                                             #
  ###############################################################
  
  data = readData(fileName)
  human = pd.read_csv('INSERT CASE FILE HERE')
  human = human.filter(items = ['d2H', 'd18O']) #Specify isotope columns in case data

  indices = [1, 2, 3, 6, 7] # Specify colums for lat, long, region, H, O
  regions, data = filterData(data, indices, target=3)

  full_X = np.array([row[3:] for row in data], dtype='float64') # Use H and O as input
  full_Y = np.array([row[2] for row in data], dtype='float64')  # To try and predict region
  
  for row in human:
      clf.fit(full_X, full_Y) # Fit whole dataset to desired classifier
      case = clf.predict(human) # Use classifier to predict the region for a casea
      print('single human prediction', human, case, clf.predict_proba(human)) 
