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
from sklearn.metrics import f1_score

np.random.seed(1234)

def get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, clf):
	
	top_class_count = 0
	for label in y_test:
		if label == most_common_classes[0]:
			top_class_count += 1
	
	print("Baseline accuracy: {0}", format(float(top_class_count) / float(y_test.shape[0])))
	y_pred = clf.predict(X_test)	
	return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='macro')

def fine_tune_parameters_given_clf(clf_name, X_train, y_train, X_test, y_test):
	
	#scores = ['precision', 'recall']
	scores = ['accuracy']#, 'f1_macro', 'f1_micro']
	
	if clf_name == "SVM":
		tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
						 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
						 {'kernel': ['poly'],
                             'degree': [1, 2, 3, 4],
                             'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
						{'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

		clf = SVC()
		
	elif clf_name == "Nearest Neighbors":
		tuned_parameters = {"n_neighbors": [1, 3, 5, 10, 20]}
		clf = KNeighborsClassifier()
		
	elif clf_name == "Decision Tree":
		tuned_parameters = {'max_depth': [i for i in range(1,33)], 
		'min_samples_split': list(np.linspace(0.1,1,10)),
		'min_samples_leaf': list(np.linspace(0.1,0.5,5)),
                  'max_features': [i for i in range(1, 10)]}
		clf = DecisionTreeClassifier()
		
	elif clf_name == "Random Forest":
		tuned_parameters = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'criterion': ['gini', 'entropy'],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 "n_estimators": [250, 500, 1000]}
		clf = RandomForestClassifier()
	
	"""
	elif clf_name == "AdaBoost":
		tuned_parameters = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]
             }
		clf = AdaBoostClassifier()
	
	elif clf_name == "MLP":
		tuned_parameters = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
		clf = MLPClassifier()
		
	elif clf_name == "Gaussian Process":
		
		clf = GaussianProcessClassifier()
	
	elif clf_name == "QDA":
		tuned_parameters = 
		clf = QuadraticDiscriminantAnalysis()
	"""
		
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()

		clf = GridSearchCV(clf, tuned_parameters, cv=4,
						   scoring='%s' % score, verbose=0)
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
	
def feature_selection(X_train, X_test, y_train):
		
	from sklearn.feature_selection import VarianceThreshold
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2
	from sklearn.feature_selection import SelectFromModel
	from sklearn.feature_selection import RFECV
	from sklearn.svm import SVC
	from sklearn.svm import LinearSVC
	from sklearn.model_selection import StratifiedKFold
	
	print("Before feature selection - X_train:{0}, X_test:{1}".format(X_train.shape, X_test.shape))	
	
	"""
	# Variance Threshold feature selection
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	X_train = sel.fit_transform(X_train)
	#print(X_train.shape)
	feature_mask = sel.get_support()
	X_test_new = np.zeros((X_test.shape[0], X_train.shape[1]))
	#print(feature_mask)
	for i in range(X_test.shape[0]):
		count = 0
		for j in range(X_test.shape[1]):
			if feature_mask[j] == True:
				X_test_new[i][count] = X_test[i][j]
				count += 1
	
	
	# Univariate feature selection
	sel = SelectKBest(chi2, k=2)
	X_train = sel.fit_transform(X_train, y_train)
	feature_mask = sel.get_support()
	X_test_new = np.zeros((X_test.shape[0], X_train.shape[1]))
	for i in range(X_test.shape[0]):
		count = 0
		for j in range(X_test.shape[1]):
			if feature_mask[j] == True:
				X_test_new[i][count] = X_test[i][j]
				count += 1
		
	# Recursive cross-validated feature elimination
	svc = SVC(kernel="linear")
	sel = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5),
				  scoring='accuracy')
	sel.fit(X_train, y_train)
	feature_mask = sel.get_support()
	number_of_features = list(feature_mask).count(True)
	X_train_new = np.zeros((X_train.shape[0], number_of_features))
	X_test_new = np.zeros((X_test.shape[0], number_of_features))
	for i in range(X_test.shape[0]):
		count = 0
		for j in range(X_test.shape[1]):
			if feature_mask[j] == True:
				X_test_new[i][count] = X_test[i][j]
				count += 1
				
	for i in range(X_train.shape[0]):
		count = 0
		for j in range(X_train.shape[1]):
			if feature_mask[j] == True:
				X_train_new[i][count] = X_train[i][j]
				count += 1
	"""
	
	# L1-norm based feature selection
	svc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
	sel = SelectFromModel(svc, prefit=True)
	X_train = sel.transform(X_train)
	feature_mask = sel.get_support()
	X_test_new = np.zeros((X_test.shape[0], X_train.shape[1]))
	for i in range(X_test.shape[0]):
		count = 0
		for j in range(X_test.shape[1]):
			if feature_mask[j] == True:
				X_test_new[i][count] = X_test[i][j]
				count += 1
	
	X_test = X_test_new
	print("After feature selection - X_train:{0}, X_test:{1}".format(X_train.shape, X_test.shape))
	
	return X_train, X_test
		
def tuned_parameters_5_fold(poi_ids, conn, args):
	
	# Shuffle ids
	poi_ids = poi_ids['poi_id']
	random.shuffle(poi_ids)
	
	kf = KFold(n_splits = 5)
	
	count = 1
	
	clf_names_not_tuned = ["Naive Bayes", "MLP", "Gaussian Process", "QDA", "AdaBoost"]
	clf_names = ["Nearest Neighbors", "SVM", "Decision Tree", "Random Forest", "AdaBoost", "Naive Bayes", "MLP", "Gaussian Process", "QDA"]
	#clf_scores_dict = dict.fromkeys(clf_names)
	#for item in clf_scores_dict:
	#	clf_scores_dict[item] = [[], [], []]
	
	# split data into train, test
	for train_ids, test_ids in kf.split(poi_ids):
		
		train_ids = [train_id + 1 for train_id in train_ids]
		test_ids = [test_id + 1 for test_id in test_ids]
		
		# get train and test sets
		X_train, y_train, X_test, y_test = get_train_test_sets(conn, args, train_ids, test_ids)
		
		#X_train, X_test = feature_selection(X_train, X_test, y_train)
		
		most_common_classes = find_10_most_common_classes_train(y_train)
		
		for clf_name in clf_names:
			print(clf_name)
			
			if clf_name in clf_names_not_tuned:
				if clf_name == "Naive Bayes":
					clf = GaussianNB()
					clf.fit(X_train, y_train)
				elif clf_name == "MLP":
					clf = MLPClassifier()
					clf.fit(X_train, y_train)
				elif clf_name == "Gaussian Process":
					clf = GaussianProcessClassifier()
					clf.fit(X_train, y_train)
				#elif clf_name == "QDA":
				#	clf = QuadraticDiscriminantAnalysis()
				#	clf.fit(X_train, y_train)
				else:
					clf = AdaBoostClassifier()
					clf.fit(X_train, y_train)
			else:
				clf = fine_tune_parameters_given_clf(clf_name, X_train, y_train, X_test, y_test)
			
			#score = clf.score(X_test, y_test)
			accuracy, f1_score_micro, f1_score_macro = get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, clf)
			#clf_scores_dict[clf_name].append()
			print("Test Accuracy Score of {0} classifier for fold number {1}: {2}".format(clf_name, count, accuracy))
			print("Micro F1 Score of {0} classifier for fold number {1}: {2}".format(clf_name, count, f1_score_micro))
			print("Macro F1 Score of {0} classifier for fold number {1}: {2}".format(clf_name, count, f1_score_macro))
			
		count += 1
		
	#for clf_name in clf_names:	
	#	print("Mean Test Accuracy Score of {0} classifier across folds: {1}". format(clf_name, sum(map(float,clf_scores_dict[clf_name])) / 5.0))
				
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-pois_tbl_name", "--pois_tbl_name", required=True,
		help="name of table containing pois information")
	ap.add_argument("-roads_tbl_name", "--roads_tbl_name", required=True,
		help="name of table containing roads information")
	ap.add_argument("-threshold_pois", "--threshold_pois", required=True,
		help="threshold for distance-specific features between pois")
	ap.add_argument("-threshold_streets", "--threshold_streets", required=True,
		help="threshold for distance-specific features between streets")
	ap.add_argument("-k_ngrams", "--k_ngrams", required=True,
		help="the number of the desired top-k most frequent ngrams")
	ap.add_argument("-k_tokens", "--k_tokens", required=True,
		help="the number of the desired top-k most frequent tokens")
	ap.add_argument("-n", "--n", required=True,
		help="the n-gram size")
	ap.add_argument("-n_tokens", "--n_tokens", required=True,
		help="the n-token size")
	ap.add_argument("-level", "--level", required=True,
		help="the class label level")
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
