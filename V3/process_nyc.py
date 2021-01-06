# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:25:17 2021

Process New York City Uber driver data.

@author: Nico Miguel, Sarah Li
"""
import pandas as pd
import decimal
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import math
import util.trip as trip
#%% DATA %%#
# Import data into numpy array
file_name = "nyc_uber\uber-raw-data-apr14.csv"
new_york_apr_14 = pd.read_csv(file_name, header = 0).to_numpy()


#%% TRIPS %%#
# Create list of trip objects
apr_14_trips = []
lat, long = 0,0
lat_limits = (9999, -9999)  # (min, max)
lng_limits = (9999, -9999)  # (min, max)
conc_trips = []
for i in range(new_york_apr_14.shape[0]):
    apr_14_trips.append(trip.Trip(new_york_apr_14[i]))

    lat_limits[1] = max(apr_14_trips[i].lat, lat_limits[1])
    lat_limits[0] = min(apr_14_trips[i].lat, lat_limits[0])
    lng_limits[1] = max(apr_14_trips[i].long, lng_limits[1])
    lng_limits[0] = min(apr_14_trips[i].long, lng_limits[0])

    if location_compare(apr_14_trips[i].long, apr_14_trips[i].lat, 
                        -73.9875, 40.75, .04, .075):
        conc_trips.append(Trip(new_york_apr_14[i]))

#%% LOCATION HISTOGRAM %%#
division = 500
location_histogram, long_edges, lat_edges = ( 
    np.histogram2d([(trips.long) for trips in apr_14_trips],
                   [(trips.lat) for trips in apr_14_trips], 
                   bins = [int((lng_limits[1]-lng_limits[0]])*division),
                           int((lat_limits[1]-lat_limits[0]])*division)], 
                   range = [[lng_limits[0], lng_limits[1]],
                            [lat_limits[0], lat_limits[1]]]))

location_histogram = location_histogram.T
        
# Split into quadrants
n_sections = 9
x_div = int(location_histogram.shape[1]/int(np.sqrt(n_sections)))
y_div = int(location_histogram.shape[0]/int(np.sqrt(n_sections)))
q1 = location_histogram[:x_div*1,:y_div*1]
long1, lat1 = long_edges[:x_div,],lat_edges[:y_div]
q2 = location_histogram[x_div*1:x_div*2,:y_div*1]
long2, lat2 = long_edges[x_div*1:x_div*2],lat1
q3 = location_histogram[y_div*2:y_div*3,:x_div*1]
long3, lat3 = long_edges[x_div*2:x_div*3],lat1
q4 = location_histogram[:x_div*1,y_div*1:y_div*2]
long4, lat4 = long1, lat_edges[y_div*1:y_div*2]
q5 = location_histogram[x_div*1:x_div*2,y_div*1:y_div*2]
long5, lat5 = long2, lat4
q6 = location_histogram[y_div*2:y_div*3,x_div*1:x_div*2]
long6, lat6 = long3, lat4
q7 = location_histogram[:x_div*1,y_div*2:y_div*3]
long7, lat7 = long1, lat_edges[y_div*2:y_div*3]
q8 = location_histogram[x_div*1:x_div*2,y_div*2:y_div*3]
long8, lat8 = long2, lat7
q9 = location_histogram[y_div*2:y_div*3,x_div*2:x_div*3]
long9, lat9 = long3, lat7

loc_hist_fin, long_edges_final, lat_edges_final = (
    np.histogram2d([(trips.long) for trips in conc_trips],
                   [(trips.lat) for trips in conc_trips], 
                   bins = [int((0.08)*division),int((0.15)*division)], 
                   range = [[-74.0275, -73.9475],[40.675,40.825]]))
loc_hist_fin = loc_hist_fin.T

#%% PLOT %%#
fig, ax = plt.subplots(figsize=(8,5))
X_loc,Y_loc = np.meshgrid(long_edges_final,lat_edges_final)
plt.pcolormesh(X_loc,Y_loc,loc_hist_fin,cmap = cm.cool)
plt.colorbar()
