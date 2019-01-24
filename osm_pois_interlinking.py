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

def explode(indf):
	from shapely.geometry.polygon import Polygon
	from shapely.geometry.multipolygon import MultiPolygon
	
	outdf = gpd.GeoDataFrame(columns=indf.columns)
	
	for idx, row in indf.iterrows():
		if type(row.geometry) == Polygon:
			outdf = outdf.append(row,ignore_index=True)
		if type(row.geometry) == MultiPolygon:
			multdf = gpd.GeoDataFrame(columns=indf.columns)
			recs = len(row.geometry)
			multdf = multdf.append([row]*recs,ignore_index=True)
			for geom in range(recs):
				multdf.loc[geom,'geometry'] = row.geometry[geom]
			outdf = outdf.append(multdf,ignore_index=True)
	return outdf
	
def import_from_json():
	"""
	Getting a geopandas dataframe from the geojson file 
	consisting of all the building details in Athens.
	The columns of this data frame are presented in the 
	following list: ['id', 'osm_id', 'name', 'type', 'geometry']
	"""
	df = gpd.read_file('/home/nikos/Desktop/greece.poi.json')
	#print(list(df.columns.values))
	
	#for index, row in df.iterrows():
	#	print(row['name'])
	
	return df
	
#def match_toponyms(toponym1, toponym2):
	
def filter_osm_pois_with_large_area(df):
	df = df.to_crs({'init': 'epsg:3857'})
	df["area"] = df['geometry'].area
	#print(df["area"])
	df = df[df['area'] < 3000.0]
	#print(df["area"])
	
	return df

def get_poi_ids_names_geometry(indf, conn, args):
	
	from shapely.geometry import Polygon
	import json
	
	# get all poi details
	#sql = "select id, geom as geom from {0}".format(args["pois_tbl_name"])
	#df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	outdf = pd.DataFrame(columns = ['poi_id', 'name', 'geom', 'polygon'])
	
	for index1, row1 in indf.iterrows():
		for geometry in list(row1['geometry'].geoms):
			coordinate_list = []
			long_list = []
			lat_list = []
			for coordinates in geometry.exterior.coords:
				coordinate_list.append(list(coordinates))
				lat_list.append(coordinates[1])
				long_list.append(coordinates[0])
			#print(coordinate_list)

			polygon = {"type":"Polygon", "coordinates":coordinate_list, "crs":{"type":"name","properties":{"name":"EPSG:3857"}}}
			polygon = json.dumps(polygon)

			if row1['name'] == None:
				sql = "select {0}.id as poi_id, {0}.name_u as name, {0}.geom as geom from {0} where ST_Intersects(ST_GeomFromGeoJSON('{1}'),  ST_LineFromMultiPoint(geom))".format(args["pois_tbl_name"], polygon)
				tempdf = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
				if not tempdf.empty:
					tempdf['geom_polygon'] = Polygon(polygon)
					outdf = outdf.append(tempdf)
					#print(outdf)
			else:
				name = row1['name']
				#if "Βασιλόπ" in name:
				#	print(name)
				
				if "'" in name:
					name = name.replace("'", "''")
				
				sql = "select {0}.id as poi_id, {0}.name_u as name, {0}.geom as geom from {0} where ST_Covers(ST_GeomFromGeoJSON('{2}'), ST_LineFromMultiPoint(ST_Transform({0}.geom, 3857))) and {0}.name_u like '%{1}%'".format(args["pois_tbl_name"], name, polygon)
				#sql = "select {0}.id as poi_id, {0}.name_u as name, {0}.geom as geom from {0} where {0}.name_l like '%{1}%'".format(args["pois_tbl_name"], name)
				tempdf = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
				if not tempdf.empty:
					tempdf['polygon'] = Polygon(zip(lat_list, long_list))
					outdf = outdf.append(tempdf)
				#print(outdf)  
	
	print(outdf)   
	#return df
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-pois_tbl_name", "--pois_tbl_name", required=True,
		help="name of table containing pois information")
	args = vars(ap.parse_args())
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	#db_pois_df = get_poi_ids_names_geometry(conn, args)
	osm_pois_df = import_from_json()
	#osm_pois_df = explode(osm_pois_df)
	#print(osm_pois_df)
	osm_pois_df = filter_osm_pois_with_large_area(osm_pois_df)
	get_poi_ids_names_geometry(osm_pois_df, conn, args)

if __name__ == "__main__":
   main()	

	
