#!/usr/bin/python

# import the necessary packages
import pandas as pd
import numpy as np
from geopy.geocoders import *
from geopy.geocoders import Nominatim

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

indf = pd.read_csv('/home/nikos/Desktop/Datasets/gc_data_1/GROUPAMA2017_GC.csv')
geolocators = [ArcGIS(timeout = 100), Nominatim(timeout = 100), Photon(timeout = 100), Yandex(timeout = 100)]
geolocator_names = ['ArcGIS', 'Nominatim', 'Photon', 'Yandex']
#print(indf)
indf = indf.dropna(subset=['Address_St', 'CITY', 'Address_Nu'])
#print(indf)
for geolocator_name in geolocator_names:
	lon_row_name = geolocator_name + '_lon'
	lat_row_name = geolocator_name + '_lat'
	indf[lon_row_name] = 0.0
	indf[lat_row_name] = 0.0

"""
dictionary = dict.fromkeys(list(indf.columns.values))
for key in dictionary:
	dictionary[key] = []

outdf = pd.DataFrame(dictionary)	
print(outdf)
"""
count = 0
count2 = 0
for index, row in indf.iterrows():
	
	if count2 > 3527:
		if hasNumbers(row['Address_Nu']):
			address = row["Address_St"] + " " + row["Address_Nu"] + ", " + row["CITY"]
			#print(address)
			geolocator_count = 0
			for geolocator, geolocator_name in zip(geolocators, geolocator_names):
				location = geolocator.geocode(address)
				if location is not None:
					geolocator_count += 1
				
			if geolocator_count == 4:
				for geolocator, geolocator_name in zip(geolocators, geolocator_names):
					location = geolocator.geocode(address)
					lat_row_name = geolocator_name + '_lat'
					lon_row_name = geolocator_name + '_lon'
					indf.loc[index, lat_row_name] = location.latitude
					indf.loc[index, lon_row_name] = location.longitude
					#print(location, location.latitude, location.longitude)
				count += 1

		if count == 100:
			print(index)
			break
	count2 += 1
		

#print(indf)
indf = indf[indf.iloc[:,19] > 0.0]
#print(indf)
indf.to_csv("geocoding.csv", mode = 'a', sep=',')
