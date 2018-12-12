#!/usr/bin/python

import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
from pois_feature_extraction import *
from textual_feature_extraction import *
from geospatial_feature_extraction import *
from feml import *
import nltk

def get_train_test_poi_ids(conn, args):
	
	from sklearn.model_selection import train_test_split
	
	# get all poi details
	sql = "select {0}.id as poi_id, {0}.geom from {0}".format(args["pois_tbl_name"])
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	X = np.asarray(df['poi_id'])
	y = np.zeros(len(df['poi_id']))
	
	poi_ids_train, poi_ids_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        
	return poi_ids_train, poi_ids_test
	
def get_poi_ids(conn, args):
	
	from sklearn.model_selection import train_test_split
	
	# get all poi details
	sql = "select {0}.id as poi_id, {0}.geom from {0}".format(args["pois_tbl_name"])
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	#ids = np.asarray(df['poi_id'])
        
	return df
        
def get_train_test_sets(conn, args, poi_ids_train, poi_ids_test):
	
	# we build a dictionary containing the poi ids as keys
	# and we map to it its x, y coordinates
	poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict(conn, args)
	
	# we read the different labels
	class_codes_set = get_class_codes_set()
	
	# we encode them so we can have a more compact representation of them
	poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict(class_codes_set, poi_id_to_class_code_coordinates_dict)
	
	y_train = []
	y_test = []
	
	X_train = []
	X_test = []
	
	#print(poi_ids_train)
	#print(poi_ids_test)
		
	for poi_id in poi_ids_train:
		#print(poi_id)
		#print(poi_id_to_encoded_labels_dict[poi_id][0][0])
		y_train.append(poi_id_to_encoded_labels_dict[poi_id][0][0])
	for poi_id in poi_ids_test:
		#print(poi_id)
		#print(poi_id_to_encoded_labels_dict[poi_id][0][0])
		y_test.append(poi_id_to_encoded_labels_dict[poi_id][0][0])
		
	y_train = np.asarray(y_train)
	y_test = np.asarray(y_test)	
		
	poi_id_to_word_features = get_features_top_k(poi_ids_train, conn, args)
	poi_id_to_word_features_ngrams = get_features_top_k_ngrams(poi_ids_train, conn, args)
	closest_pois_boolean_and_counts_per_label = get_closest_pois_boolean_and_counts_per_label(poi_ids_train, conn, args, float(args["threshold"]))
	closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(poi_ids_train, conn, args, float(args["threshold"]))
	
	for poi_id in poi_ids_train:
		temp_feature_list1 = [item for sublist in closest_pois_boolean_and_counts_per_label[poi_id] for item in sublist]
		temp_feature_list2 = [item for sublist in closest_pois_boolean_and_counts_per_label_streets[poi_id] for item in sublist]
		feature_list = poi_id_to_word_features[poi_id] + poi_id_to_word_features_ngrams[poi_id] + temp_feature_list1 + temp_feature_list2
		X_train.append(feature_list)
	
	poi_id_to_word_features = get_features_top_k(poi_ids_test, conn, args)
	poi_id_to_word_features_ngrams = get_features_top_k_ngrams(poi_ids_test, conn, args)
	closest_pois_boolean_and_counts_per_label = get_closest_pois_boolean_and_counts_per_label(poi_ids_test, conn, args, float(args["threshold"]))
	closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(poi_ids_test, conn, args, float(args["threshold"]))
	
	for poi_id in poi_ids_test:
		temp_feature_list1 = [item for sublist in closest_pois_boolean_and_counts_per_label[poi_id] for item in sublist]
		temp_feature_list2 = [item for sublist in closest_pois_boolean_and_counts_per_label_streets[poi_id] for item in sublist]
		feature_list = poi_id_to_word_features[poi_id] + poi_id_to_word_features_ngrams[poi_id] + temp_feature_list1 + temp_feature_list2
		X_test.append(feature_list)

	X_train = np.asarray(X_train)
	X_test = np.asarray(X_test)
	
	print(X_train.shape)
	print(X_test.shape)
		
	return X_train, y_train, X_test, y_test

def standardize_data(X_train, X_test):
	from sklearn.preprocessing import StandardScaler
	
	standard_scaler = StandardScaler()
	X_train = standard_scaler.fit_transform(X_train)
	X_test = standard_scaler.transform(X_test)
	
	return X_train, X_test
	
def find_10_most_common_classes_train(y_train):
	
	labels = list(y_train)
	
	label_counter = {}
	for label in labels:
		if label in label_counter:
			label_counter[label] += 1
		else:
			label_counter[label] = 1

	classes_ranked = sorted(label_counter, key = label_counter.get, reverse = True) 
	most_common_classes = classes_ranked[:10]
	
	return most_common_classes

