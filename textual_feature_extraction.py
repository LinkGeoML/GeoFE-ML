#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
from feml import *
import nltk

def find_ngrams(token_list, n):
	s = []
    
	#for token in token_list:
	#	for i in range(0, len(token)- n + 1):
	#		s.append(token[i:n+i])
	
	#for i in range(len(token_list) - n + 1):
	#	s.append(token_list[i] + " " + token_list[i+1])
	
	for i in range(len(token_list)):
		if i == 0:
			s.append(token_list[i] + " ")
		elif i == len(token_list) - 1:
			s.append(" " + token_list[i])
		else:
			s.append(" " + token_list[i])
			s.append(token_list[i] + " ")
	
	return s

def get_corpus(ids, conn, args, n_grams = False):
	
	""" This function queries the names of all the pois in the dataset
		and creates a corpus from the words in them"""
	
	#nltk.download()
	
	# get all poi details
	
	"""
	count = 0
	for id in ids:
		sql = "select {0}.id as poi_id, {0}.class_code, {0}.geom from {0} where {0}.id = {1}".format(args["pois_tbl_name"], id)
		if count == 0:
			df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
		else:
			temp_df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
			frames = [df, temp_df]
			df = gpd.GeoDataFrame( pd.concat( frames, ignore_index=True) )
		count += 1
	"""
	sql = "select {0}.id as poi_id, {0}.class_code, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	corpus = []
	
	# for every poi name
	for index, row in df.iterrows():
		# perform stemming based on the language it's written in
		stemmed_word = perform_stemming(row['name'], lang_detect=True)
		# break it in tokens
		not_stopwords, stopwords = normalize_str(row['name'])
		not_stopwords = list(not_stopwords)
		
		if n_grams:
			not_stopwords = find_ngrams(not_stopwords, int(args["n"]))		
		corpus.append(not_stopwords)
		
	corpus = [elem for sublist in corpus for elem in sublist]
	
	if not n_grams:
		corpus = [elem for elem in corpus if len(elem) > 2]
	
	#print(corpus)
	
	return corpus
	
def get_top_k_features(corpus, args):
	word_counter = {}
	
	for word in corpus:
		if word in word_counter:
			word_counter[word] += 1
		else:
			word_counter[word] = 1
			
	popular_words = sorted(word_counter, key = word_counter.get, reverse = True)	
	
	
	import csv
	
	"""
	with open('word_features_ranked.csv', 'a') as csvFile:
		writer = csv.writer(csvFile)
		for word in popular_words:
			list_to_csv = [word, word_counter[word]]
				
			writer.writerow(list_to_csv)

	csvFile.close()
	"""

	#print(popular_words)
	
	top_k = popular_words[:int(args["k"])]
	
	return top_k

def get_poi_top_k_features(ids, conn, top_k_features, args):
	# get all poi details
	
	"""
	count = 0
	for id in ids:
		sql = "select {0}.id as poi_id, {0}.class_code, {0}.geom from {0} where {0}.id = {1}".format(args["pois_tbl_name"], id)
		if count == 0:
			df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
		else:
			temp_df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
			frames = [df, temp_df]
			df = gpd.GeoDataFrame( pd.concat( frames, ignore_index=True) )
		count += 1
	"""
	sql = "select {0}.id as poi_id, {0}.class_code, {0}.geom from {0} where {0}.id in {1}".format(args["pois_tbl_name"], tuple(ids))
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	poi_id_to_boolean_top_k_features_dict = dict.fromkeys(df['poi_id'])
	for poi_id in poi_id_to_boolean_top_k_features_dict:
		poi_id_to_boolean_top_k_features_dict[poi_id] = [0 for _ in range(0, int(args["k"]))]
	
	for index, row in df.iterrows():
		 for i in range(len(top_k_features)):
			 if top_k_features[i] in row['name'].lower():
				 poi_id_to_boolean_top_k_features_dict[row['poi_id']][i] = 1
				 
	return poi_id_to_boolean_top_k_features_dict

def get_features_top_k(ids, conn, args):
	""" This function extracts frequent terms from the whole corpus of POI names. 
		During this process, it optionally uses stemming. Selects the top-K most 
		frequent terms and creates feature positions for each of these terms."""
		
	# get corpus
	corpus = get_corpus(ids, conn, args)
	
	# find top k features
	top_k_features = get_top_k_features(ids, corpus, args)
		
	# get boolean values dictating whether pois have or haven't any of the top features in their names
	return get_poi_top_k_features(conn, top_k_features, args)
	
def get_features_top_k_ngrams(ids, conn, args):
	""" This function extracts frequent n-grams (n is specified) from the whole 
	corpus of POI names. It selects the top-K most frequent n-grams and creates
	feature positions for each of these terms."""
	
	# get corpus
	corpus = get_corpus(ids, conn, args, n_grams = True)
	
	# find top k features
	top_k_features = get_top_k_features(corpus, args)
	
	# get boolean values dictating whether pois have or haven't any of the top features in their names
	return get_poi_top_k_features(ids, conn, top_k_features, args)
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-pois_tbl_name", "--pois_tbl_name", required=True,
		help="name of table containing pois information")
	ap.add_argument("-k", "--k", required=True,
		help="the number of the desired top-k most frequent tokens")
	ap.add_argument("-n", "--n", required=True,
		help="the n-gram size")
	args = vars(ap.parse_args())
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	features_top_k_words = get_features_top_k(conn, args)
	
	#print(features_top_k_words)
	
	#features_top_k_ngrams = get_features_top_k_ngrams(conn, args)
	#print(features_top_k_ngrams)
	
	"""
	import csv
	
	with open('data.csv', 'a') as csvFile:
		writer = csv.writer(csvFile)
		for id in features_top_k_ngrams:
			writer.writerow(features_top_k_ngrams[id])

	csvFile.close()
	"""
if __name__ == "__main__":
   main()
