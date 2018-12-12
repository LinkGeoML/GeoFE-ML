#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
from pois_feature_extraction import *
from textual_feature_extraction import *
from feml import *
import nltk
import itertools
import random

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

np.random.seed(1234)

def get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, clf):
	y_pred = clf.predict(X_test)
	
	count = 0
	most_common_classes_count = 0
	
	for truth, pred in zip(y_test, y_pred):
		if truth in most_common_classes:
			if truth == pred:
				count += 1
				
			most_common_classes_count += 1
	
	return float(count) / float(most_common_classes_count)

def fine_tune_parameters_given_clf(clf_name):
	
	#scores = ['precision', 'recall']
	scores = ['accuracy']
	
	if clf_name == "SVM":
		tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
						 'C': [1, 10, 100, 1000]},
						 {'kernel': ['poly'],
                             'degree': [1, 2, 3, 4],
                             'C': [1, 10, 100, 1000]},
						{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

		clf = SVC()
		
	elif clf_name == "Nearest Neighbors":
		tuned_parameters = {"n_neighbors": [1, 3, 5, 10, 20]}
		clf = KNeighborsClassifier()
		
	elif clf_name == "Decision Tree":
		tuned_parameters = {'max_depth': [1, 2, 3, 4, 5],
                  'max_features': [1, 2, 3, 4]}
		clf = DecisionTreeClassifier()
		
	elif clf_name == "Random Forest":
		tuned_parameters = {"n_estimators": [250, 500, 1000]}
		clf = RandomForestClassifier()
		
	elif clf_name == "AdaBoost":
		tuned_parameters = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]
             }
		clf = AdaBoostClassifier()
		
	#elif clf_name == "MLP":
	#	tuned_parameters = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
	#	clf = MLPClassifier()
		
	#elif clf_name == "Gaussian Process":
	#	
	#	clf = GaussianProcessClassifier()
	
	#elif clf_name == "QDA":
	#	tuned_parameters = 
	#	clf = QuadraticDiscriminantAnalysis()
		
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()

		clf = GridSearchCV(clf, tuned_parameters, cv=4,
						   scoring='%s_macro' % score)
		clf.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r"
				  % (mean, std * 2, params))
		print()
		
	return clf
		
def default_parameters_5_fold(X_train, X_test, y_train, y_test):
	
	X = list(np.concatenate((X_train, X_test), axis=0))
	y = list(np.concatenate((y_train, y_test), axis=0))
	
	c = list(zip(X, y))

	random.shuffle(c)

	X, y = zip(*c)
	
	X = np.asarray(X)
	y = np.asarray(y)
	
	kf = KFold(n_splits=5)
	count = 1
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		print("Displaying results for fold {0}".format(count))
		
		names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Naive Bayes"]

		classifiers = [
			KNeighborsClassifier(3),
			SVC(kernel="linear", C=0.025),
			SVC(gamma=2, C=1),
			DecisionTreeClassifier(max_depth=5),
			RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
			GaussianNB()]
		
		# iterate over classifiers and produce results
		for name, clf in zip(names, classifiers):
			clf.fit(X_train, y_train)
			score = clf.score(X_test, y_test)
			print("Accuracy Score of {0} classifier: {1}".format(name, score))
			
		count += 1
		
def tuned_parameters_5_fold(poi_ids, conn, args):
	
	# Shuffle ids
	poi_ids = poi_ids['poi_id']
	random.shuffle(poi_ids)
	
	kf = KFold(n_splits = 5)
	
	count = 1
	
	clf_names = ["Nearest Neighbors", "SVM", "Decision Tree", "Random Forest", "Naive Bayes", "MLP", "Gaussian Process", "AdaBoost", "QDA"]
	clf_scores_dict = dict.fromkeys(clf_names)
	for item in clf_scores_dict:
		clf_scores_dict[item] = []
	
	# split data into train, test
	for train_ids, test_ids in kf.split(poi_ids):
		
		# get train and test sets
		X_train, y_train, X_test, y_test = get_train_test_sets(conn, args, train_ids, test_ids)
		
		most_common_classes = find_10_most_common_classes_train(y_train)
		
		for clf_name in clf_names:
			clf = fine_tune_parameters_given_clf(clf_name)
			
			#score = clf.score(X_test, y_test)
			score = get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, clf)
			clf_scores_dict[clf_name].append(score)
			print("Test Accuracy Score of {0} classifier for fold number {1}: {2}".format(clf_name, count, score))
			
		count += 1
		
	for clf_name in clf_names:	
		print("Mean Test Accuracy Score of {0} classifier across folds: {1}". format(clf_name, sum(map(float,clf_scores_dict[clf_name])) / 5.0))
				
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-pois_tbl_name", "--pois_tbl_name", required=True,
		help="name of table containing pois information")
	ap.add_argument("-roads_tbl_name", "--roads_tbl_name", required=True,
		help="name of table containing roads information")
	ap.add_argument("-threshold", "--threshold", required=True,
		help="threshold for distance-specific features")
	ap.add_argument("-k", "--k", required=True,
		help="the number of the desired top-k most frequent tokens")
	ap.add_argument("-n", "--n", required=True,
		help="the n-gram size")
	args = vars(ap.parse_args())
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	# get the poi ids
	poi_ids = get_poi_ids(conn, args)
	tuned_parameters_5_fold(poi_ids, conn, args)
	
	# get the data
	
	#poi_ids_train, poi_ids_test = get_train_test_poi_ids(conn, args)
	#X_train, y_train, X_test, y_test = get_train_test_sets(conn, args, poi_ids_train, poi_ids_test)
	
	# save it in csv files
	
	#np.savetxt("train.csv", X_train, delimiter=",")
	#np.savetxt("test.csv", X_test, delimiter=",")
	#np.savetxt("labels_train.csv", y_train, delimiter=",")
	#np.savetxt("labels_test.csv", y_test, delimiter=",")
	
	# load it from csv files
	#X_train = np.genfromtxt('train.csv', delimiter=',')
	#X_test = np.genfromtxt('test.csv', delimiter=',')
	#y_train = np.genfromtxt('labels_train.csv', delimiter=',')
	#y_test = np.genfromtxt('labels_test.csv', delimiter=',')
	
	#default_parameters_single_fold(X_train, X_test, y_train, y_test)
	
	#default_parameters_5_fold(X_train, X_test, y_train, y_test)
	
if __name__ == "__main__":
   main()
