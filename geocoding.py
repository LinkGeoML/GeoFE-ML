#!/usr/bin/python

# import the necessary packages
import pandas as pd
import numpy as np
from geopy.geocoders import *
from geopy.geocoders import Nominatim
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
from feml import *
import nltk
import fiona
from shapely.geometry import mapping, Point, MultiPoint

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

"""
indf = pd.read_csv('/home/nikos/Desktop/Datasets/gc_data_1/GROUPAMA2017_GC.csv')
geolocators = [ArcGIS(timeout = 100), Nominatim(timeout = 100), Photon(timeout = 100), Yandex(timeout = 100)]
geolocator_names = ['ArcGIS', 'Nominatim', 'Photon', 'Yandex']
indf = indf.dropna(subset=['Address_St', 'Address_Ci', 'Address_Nu', "Address_PC"])
for geolocator_name in geolocator_names:
	lon_row_name = geolocator_name + '_lon'
	lat_row_name = geolocator_name + '_lat'
	indf[lon_row_name] = 0.0
	indf[lat_row_name] = 0.0

count = 0
#count2 = 0
for index, row in indf.iterrows():
	
	if index > 41912:
		#	print(row['Address_Nu'])
		if hasNumbers(row['Address_Nu']):
			#print(row['Address_Nu'])
			address = row["Address_St"] + " " + row["Address_Nu"] + ", " + row["Address_Ci"] + ", " + str(int(row["Address_PC"]))
			
			
			#print(address)
			#geolocator_count = 0
			#for geolocator, geolocator_name in zip(geolocators, geolocator_names):
				#location = geolocator.geocode(address)
				#print(geolocator_name)
				#print(location)
				#if location is not None:
				#	geolocator_count += 1
			
			#if geolocator_count == 4:
			#	print("edw")
			
			for geolocator_name in geolocator_names:
				if geolocator_name == 'ArcGIS':
					geolocator = ArcGIS()
				elif geolocator_name == 'Nominatim':
					geolocator = Nominatim()
				elif geolocator_name == 'Photon':
					geolocator = Photon()
				else:
					geolocator = Yandex()
				location = geolocator.geocode(address, timeout = 100)
				lat_row_name = geolocator_name + '_lat'
				lon_row_name = geolocator_name + '_lon'
				if location is not None:
					indf.loc[index, lat_row_name] = location.latitude
					indf.loc[index, lon_row_name] = location.longitude
					#print(location, location.latitude, location.longitude)
				else:
					indf.loc[index, lat_row_name] = np.NaN
					indf.loc[index, lon_row_name] = np.NaN
			count += 1

			if count == 500:
				print(index, count)
				break
	#count2 += 1
#print(index, count2)
"""

#print(indf)
#indf = indf[indf.iloc[:,19] > 0.0]
#print(indf)
#indf.to_csv("geocoding_nas.csv", mode = 'a', header = False, sep=',')


indf = pd.read_csv('/home/nikos/Desktop/Working Code/geocoding_1000records.csv')
#indf = indf.sample(frac = 1)
#indf.to_csv("geocoding-shuffled.csv", sep=',')

#df = pd.read_csv('/home/nikos/Desktop/Working Code/geocoding_nas.csv')
#df = df.dropna(subset = ['ArcGIS_lon', 'ArcGIS_lat', 'Nominatim_lon', 'Nominatim_lat',
# 'Photon_lon', 'Photon_lat', 'Yandex_lon', 'Yandex_lat'])
#df = df.drop_duplicates(subset = ['Address_St', 'Address_Nu', 'Address_PC', 'Address_Ci'])
#df.to_csv("geocoding_6500records.csv", sep=',')
#print(df.shape)
"""
df = pd.read_csv('/home/nikos/Desktop/Working Code/geocoding_6500records.csv')
df = df.sample(n = 1000)
df.to_csv("geocoding_1000records.csv", sep=',')
"""

# call the appropriate function to connect to the database
conn = connect_to_db()

count = 0
for index, row in indf.iterrows():
	x, y = row['X5'], row['Y5']
	sql = "select ST_MakePoint({0}, {1}) as geom".format(x, y)
	tempdf = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	point = tempdf['geom'][0]
	
	#print(point1)
	#x, y = row['Nominatim_lon'], row['Nominatim_lat']
	#sql = "select ST_MakePoint({0}, {1}) as geom".format(x, y)
	#tempdf = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	#point2 = tempdf['geom'][0]
	#x, y = row['Photon_lon'], row['Photon_lat']
	#sql = "select ST_MakePoint({0}, {1}) as geom".format(x, y)
	#tempdf = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	#point3 = tempdf['geom'][0]
	#x, y = row['Yandex_lon'], row['Yandex_lat']
	#sql = "select ST_MakePoint({0}, {1}) as geom".format(x, y)
	#tempdf = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	#point4 = tempdf['geom'][0]
	
	#point_list = [point1, point2, point3, point4]
	
	#multipoint = MultiPoint(points = point_list)
	
	
	
	schema = {
    'geometry': 'Point',
    'properties': {'Address': 'str'},
	}
	
	if count == 0:
		with fiona.open('Χ5-Υ5.shp', 'w', 'ESRI Shapefile', schema, encoding = 'greek') as c:
		## If there are multiple geometries, put the "for" loop here
			address = row["Address_St"] + " " + row["Address_Nu"] + " " + str(row["Address_PC"]) + ", " + row["Address_Ci"]
			c.write({'geometry': mapping(point), 'properties': {'Address': address}},)
		c.close()
	else:
		with fiona.open('Χ5-Υ5.shp', 'a', 'ESRI Shapefile', schema, encoding = 'greek') as c:
		## If there are multiple geometries, put the "for" loop here
			address = row["Address_St"] + " " + row["Address_Nu"] + " " + str(row["Address_PC"]) + ", " + row["Address_Ci"]
			c.write({'geometry': mapping(point), 'properties': {'Address': address}},)
		c.close()
		
	count += 1
