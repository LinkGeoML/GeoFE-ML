#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import pandas as pd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
	
def get_poi_id_to_closest_street_id_dict(ids, conn, args):

	"""
	*** This function maps each poi to its closest road id.
	***
	*** Returns - a dictionary consisting of the poi ids as
	*** 		  its keys and their corresponding closest 
	***			  road id as its value.
	"""
		
	sql = "select {0}.id as poi_id, {0}.class_code, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	# construct a dictionary from their ids
	# also get its class_code
	poi_id_to_edge_id_dict = dict.fromkeys(df['poi_id'])
	for index, row in df.iterrows():
		poi_id_to_edge_id_dict[row['poi_id']] = [row['class_code'], 0]
		
	for index, row in df.iterrows():
		# for each poi find its distance to all the roads in the dataset
		sql = "select {0}.id as poi_id, {1}.id as edge_id, {0}.geom, ST_Distance(ST_Transform({0}.geom, 32634), {1}.geom) as dist from {0}, {1} where {0}.id = {2}".format(args["pois_tbl_name"], args["roads_tbl_name"], row['poi_id'])
		dist_df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
		
		# get the index of the minimum one and map the corresponding edge id to the poi
		distance_list = list(dist_df['dist'])
		min_index = distance_list.index(min(distance_list))
		poi_id_to_edge_id_dict[row['poi_id']][1] = dist_df['edge_id'][min_index]

	return poi_id_to_edge_id_dict

def get_street_id_to_closest_pois_boolean_and_counts_per_label_dict(ids, conn, threshold, args):
	
	"""
	*** This function maps the street ids to their closest pois' label
	*** booleans and counts. The closest pois are determined by examining
	*** whether they are located within threshold distance of the street
	"""
	
	# get all the pois
	
	if int(args['level']) == 1:
		sql = "select {0}.id as poi_id, {0}.theme as class_code, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
	elif int(args['level']) == 2:
		sql = "select {0}.id as poi_id, {0}.class_name as class_code, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
	else:
		sql = "select {0}.id as poi_id, {0}.subclass_n as class_code, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
	df_pois = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	# create a dictionary with the poi ids as its keys
	id_dict = dict.fromkeys(df_pois['poi_id'])
	for index, row in df_pois.iterrows():
		id_dict[row['poi_id']] = [row['class_code'], 0, 0]
	
	# get the class codes set and encode the class codes to labels
	class_codes_set = get_class_codes_set(args)
	id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict(class_codes_set, id_dict)
	num_of_labels = len(encoded_labels_set)
	
	# create a dictionary with the street ids as its keys
	sql = "select {0}.id as edge_id, geom from {0}".format(args["roads_tbl_name"])
	df_edges = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	street_id_to_closest_pois_boolean_and_counts_per_label_dict = dict.fromkeys(df_edges['edge_id'])
	
	# prepare the street id dictionary to be able to store the
	# boolean and count duplet for each of the class labels
	for edge_id in street_id_to_closest_pois_boolean_and_counts_per_label_dict:
		street_id_to_closest_pois_boolean_and_counts_per_label_dict[edge_id] = [[0,0] for _ in range(0, num_of_labels)]

	for index, row in df_edges.iterrows():
		# for each road find its distance to all the pois in the dataset
		count = 0
		
		sql = "select {0}.id as poi_id, {1}.id as edge_id, {0}.geom, ST_Distance(ST_Transform({0}.geom, 32634), {1}.geom) as dist from {0}, {1} where {1}.id = {2} and {0}.id in {3}".format(args["pois_tbl_name"], args["roads_tbl_name"], row['edge_id'], tuple(ids))
		dist_df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
		
		for index, row in dist_df.iterrows():
			if row['dist'] < threshold:
				street_id_to_closest_pois_boolean_and_counts_per_label_dict[row['edge_id']][id_to_encoded_labels_dict[row['poi_id']][0][0]][0] = 1
				street_id_to_closest_pois_boolean_and_counts_per_label_dict[row['edge_id']][id_to_encoded_labels_dict[row['poi_id']][0][0]][1] += 1
	
	return street_id_to_closest_pois_boolean_and_counts_per_label_dict
	
def update_poi_id_dictionary(poi_id_to_street_id_dict, street_id_to_closest_pois_boolean_and_counts_per_label_dict):
	
	"""
	*** This function just copies the contents of street_id_to_closest_pois_dict
	*** to the newly created poi_id_to_closest_pois_boolean_count_dict based on
	*** the poi_id_to_street_id_dict values which will act as keys for the 
	*** street_id_to_closest_pois_dict dictionary
	"""
	
	# construct a dictionary from the ids of the pois
	poi_id_to_closest_pois_boolean_count_dict = dict.fromkeys(poi_id_to_street_id_dict.keys(),[])
	
	# for each poi id, get its closest road id and then copy
	# this road's closest poi label boolean and count to the
	# dictionary poi_id_to_closest_pois_boolean_count_dict
	for poi_id in poi_id_to_street_id_dict:
		poi_id_to_closest_pois_boolean_count_dict[poi_id] = street_id_to_closest_pois_boolean_and_counts_per_label_dict[poi_id_to_street_id_dict[poi_id][1]]

	return poi_id_to_closest_pois_boolean_count_dict
	
	
def get_closest_pois_boolean_and_counts_per_label_streets(ids, conn, args, threshold = 1000.0):
	
	# get the dictionary mapping each poi id to that of its closest road
	poi_id_to_closest_street_id_dict = get_poi_id_to_closest_street_id_dict(ids, conn, args)
	#print(poi_id_to_closest_street_id_dict)
	
	# for every street id get the label boolean and counts values of the
	# pois located within threshold distance from it
	# (this will resemble the get_poi_id_to_boolean_and_counts_per_class_dict
	#  function but with road ids as the keys of the dictionary)
	street_id_to_label_boolean_counts_dict = get_street_id_to_closest_pois_boolean_and_counts_per_label_dict(ids, conn, threshold, args)
	
	# construct a dictionary similar to the one returned by get_poi_id_to_boolean_and_counts_per_class_dict
	# which will map a poi's id to the label boolean and count values of the poi's situated within threshold 
	# distance of the poi's closest road
	poi_id_to_closest_pois_boolean_count_dict = update_poi_id_dictionary(poi_id_to_closest_street_id_dict, street_id_to_label_boolean_counts_dict)
	return poi_id_to_closest_pois_boolean_count_dict
	
	
def get_class_codes_set(args):
	
	"""
	*** This function is responsible for reading the excel file
	*** containing the dataset labels (here stored in a more code-like
	*** manner rather than resembling labels).
	***
	*** Returns - a list of the class codes
	"""
	import pandas as pd
	
	# read the file containing the class codes
	df = pd.read_excel('/home/nikos/Desktop/Datasets/GeoData_PoiMarousi/GeoData_poiClasses.xlsx', sheet_name=None)		
	
	# store the class codes (labels) in the list
	if int(args['level']) == 1:
		class_codes = list(df['poiClasses']['THEME'])
		#print(class_codes)
	elif int(args['level']) == 2:
		class_codes = list(df['poiClasses']['CLASS_NAME'])
	else:
		class_codes = list(df['poiClasses']['SUBCLASS_N'].dropna())
		#print(class_codes)
	return class_codes
	
def get_poi_id_to_encoded_labels_dict(labels_set, id_dict):
	
	"""
	*** This function encodes our labels to values between 0 and len(labels_set)
	*** in order to have a more compact and user-friendly encoding of them.
	***
	*** Arguments - labels_set: the set of the labels (class codes) as we
	*** 			extracted them from the excel file
	***				id_dict: the dictionary containing the ids of the pois
	***
	*** Returns -	id_dict: an updated version of our pois dictionary
	***						 now mapping their ids to their encoded labels
	***				labels_set: the encoded labels set
	"""
	
	from sklearn.preprocessing import LabelEncoder
	
	# fit the label encoder to our labels set
	le = LabelEncoder()
	le.fit(labels_set)
	
	# map each poi id to its respective decoded label
	for key in id_dict:
		id_dict[key][0] = le.transform([id_dict[key][0]])
	
	return id_dict, le.transform(labels_set)
	
def get_poi_id_to_class_code_coordinates_dict(conn, args):
	
	"""
	*** This function returns a dictionary with poi ids as its keys and a 
	*** list in the form of [< poi's class code >, < x coordinate > < y coordinate >]
	*** as its values.
	"""
	# get the poi categories depending on level
	if int(args['level']) == 1:
		sql = "select {0}.id, {0}.theme, {0}.x, {0}.y, {0}.geom from {0}".format(args["pois_tbl_name"])
	elif int(args['level']) == 2:
		sql = "select {0}.id, {0}.class_name, {0}.x, {0}.y, {0}.geom from {0}".format(args["pois_tbl_name"])
	else:
		sql = "select {0}.id, {0}.subclass_n, {0}.x, {0}.y, {0}.geom from {0}".format(args["pois_tbl_name"])
	
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')

	poi_id_to_class_code_coordinates_dict = dict.fromkeys(df['id'])
	
	
	for index, row in df.iterrows():
		if int(args['level']) == 1:
			poi_id_to_class_code_coordinates_dict[row['id']] = [row['theme'], float(row['x']), float(row['y'])]
		elif int(args['level']) == 2:
			poi_id_to_class_code_coordinates_dict[row['id']] = [row['class_name'], float(row['x']), float(row['y'])]
		else:
			poi_id_to_class_code_coordinates_dict[row['id']] = [row['subclass_n'], float(row['x']), float(row['y'])]
	
	return poi_id_to_class_code_coordinates_dict
	
def get_poi_id_to_boolean_and_counts_per_class_dict(ids, conn, num_of_labels, poi_id_to_encoded_labels_dict, threshold, args):
	
	"""
	*** This function is responsible for mapping the pois to a list of two-element lists.
	*** The first element of that list will contain a  boolean value referring
	*** to whether a poi of that index's label is within threshold distance
	*** of the poi whose id is the key of this list in the dictionary. The second
	*** element contains the respective count of the pois belonging to the
	*** specific index's label that are within threshold distance of the poi-key.
	***
	*** For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	*** are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	*** id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	***
	*** Arguments - num_of_labels: the total number of the different labels
	*** 			encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	***				threshold: the aforementioned threshold
	"""
	
	from sklearn.neighbors import DistanceMetric
	from scipy.spatial import distance
	
	# define the apropriate distance metric for measuring distance between two pois
	dist = DistanceMetric.get_metric('euclidean')
	
	# get the poi ids and construct a dictionary with their ids as its keys
	count = 0
	"""
	for id in ids:
		sql = "select {0}.id as poi_id, {0}.class_code, {0}.geom, {0}.x, {0}.y from {0} where {0}.id = {1}".format(args["pois_tbl_name"], id)
		if count == 0:
			df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
		else:
			temp_df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
			frames = [df, temp_df]
			df = gpd.GeoDataFrame( pd.concat( frames, ignore_index=True) )
		count += 1
	"""
	
	#print(ids)
	if int(args['level']) == 1:
		sql = "select {0}.id as poi_id, {0}.theme as class_code, {0}.geom, {0}.x, {0}.y from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
	elif int(args['level']) == 2:
		sql = "select {0}.id as poi_id, {0}.class_name as class_code, {0}.geom, {0}.x, {0}.y from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
	else:
		sql = "select {0}.id as poi_id, {0}.subclass_n as class_code, {0}.geom, {0}.x, {0}.y from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
	#sql = "select {0}.id as poi_id, {0}.class_code, {0}.geom, {0}.x, {0}.y from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	#print(df)
	
	poi_id_to_label_boolean_counts_dict = dict.fromkeys(df['poi_id'])
	
	# add dummy values to the dictionary in order to initialize it
	# in a form that resembles its desired final form
	for poi_id in poi_id_to_label_boolean_counts_dict:
		poi_id_to_label_boolean_counts_dict[poi_id] = [[0,0] for _ in range(0, num_of_labels)]
	
	for index1, row1 in df.iterrows():
		for index2, row2 in df.iterrows():
			if row1['poi_id'] != row2['poi_id']:
				# get their coordinates
				point1 = (row1['x'], row1['y'])
				point2 = (row2['x'], row2['y'])
				# if the two points are within treshold distance, 
				# update the dictionary accordingly
				if distance.euclidean(point1, point2) < threshold:
					poi_id_to_label_boolean_counts_dict[row1['poi_id']][poi_id_to_encoded_labels_dict[row2['poi_id']][0][0]][0] = 1
					poi_id_to_label_boolean_counts_dict[row1['poi_id']][poi_id_to_encoded_labels_dict[row2['poi_id']][0][0]][1] += 1
	
	return poi_id_to_label_boolean_counts_dict
	
def get_closest_pois_boolean_and_counts_per_label(ids, conn, args, threshold = 1000.0):
	
	"""
	*** This function returns a dictionary with the poi ids as its keys
	*** and two lists for each key. The first list contains boolean values
	*** dictating whether a poi of that index's label is within threshold
	*** distance with the key poi. The second list contains the counts of
	*** the pois belonging to the same index's label.
	
	*** Arguments - threshold: we only examine pois the distance between 
	*** 			which is below the given threshold
	"""
	
	# we build a dictionary containing the poi ids as keys
	# and we map to it its x, y coordinates
	poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict(conn, args)
	
	# we read the different labels
	class_codes_set = get_class_codes_set(args)
	
	# we encode them so we can have a more compact representation of them
	poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict(class_codes_set, poi_id_to_class_code_coordinates_dict)
	
	return get_poi_id_to_boolean_and_counts_per_class_dict(ids, conn, len(encoded_labels_set), poi_id_to_encoded_labels_dict, threshold, args)

def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-pois_tbl_name", "--pois_tbl_name", required=True,
		help="name of table containing pois information")
	ap.add_argument("-roads_tbl_name", "--roads_tbl_name", required=True,
		help="name of table containing roads information")
	args = vars(ap.parse_args())
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	threshold = 1000.0
	closest_pois_boolean_and_counts_per_label = get_closest_pois_boolean_and_counts_per_label(conn, args, threshold)
	print(closest_pois_boolean_and_counts_per_label)
	
	closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(conn, args, threshold)
	print(closest_pois_boolean_and_counts_per_label_streets)
	
	"""
	import csv
	
	with open('data.csv', 'a') as csvFile:
		writer = csv.writer(csvFile)
		for id in closest_pois_boolean_and_counts_per_label:
			feature_list = [item for sublist in closest_pois_boolean_and_counts_per_label[id] for item in sublist]
				
			#print(feature_list)
			writer.writerow(feature_list)

	csvFile.close()
	"""
	
if __name__ == "__main__":
   main()
