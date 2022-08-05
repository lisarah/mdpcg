# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:27:57 2020

Purpose: Process Uber Data from trips in April 2014 in New York areas

Authors: Nicolas Miguel, Sarah Li
"""

#%% SETUP %%#

# Libraries
import pandas as pd
import numpy as np
import datetime
import decimal
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import math

# Define trip class
class Trip:
    def __init__(self, instance):
        # Date retrieval
        date_ = instance[0].split(" ")[0]
        date = date_.replace(date_[-4:], date_[-2:])
        self.date = datetime.datetime.strptime(date, '%m/%d/%y')
        
        # Time retrieval
        time_ = instance[0].split(" ")[1]
        time = time_.split(":")
        hours_ = int(time[0])
        minutes_ = int(time[1])
        seconds_ = int(time[2])
        elapsed = datetime.timedelta(hours = hours_, minutes = minutes_, seconds = seconds_)
        self.time = elapsed.seconds/3600 # Time in hours since beginning of day
        
        # Pickup Location retrieval
        self.long = instance[2]
        self.lat = instance[1]
        
        # Base
        self.base = instance[3]

# Define compare function
def location_compare(trip_long, trip_lat, area_long, area_lat, bound_long, bound_lat):
    long_check = 0
    lat_check = 0
    if abs(trip_long - area_long) <= bound_long:
        long_check = 1
    if abs(trip_lat - area_lat) <= bound_lat:
        lat_check = 1
    if long_check and lat_check:
        return 1
    else:
        return 0
    
#%% DATA %%#
# Import data into numpy array
new_york_apr_14 = pd.read_csv("D:\\Behcet\\GameTheory\\NewYorkUber\\uber-raw-data-apr14.csv", header = 0).to_numpy()


#%% TRIPS %%#
# Create list of trip objects
apr_14_trips = []
lat, long = 0,0
firsttime = True
conc_trips = []
for i in range(new_york_apr_14.shape[0]):
    apr_14_trips.append(Trip(new_york_apr_14[i]))

    if firsttime == True:
        maxLat = apr_14_trips[0].lat
        maxLong = apr_14_trips[0].long
        minLat = maxLat
        minLong = maxLong
        firsttime = False
    
    if apr_14_trips[i].lat > maxLat:
        maxLat = apr_14_trips[i].lat
    elif apr_14_trips[i].lat < minLat:
        minLat = apr_14_trips[i].lat
        
    if apr_14_trips[i].long > maxLong:
        maxLong = apr_14_trips[i].long
    elif apr_14_trips[i].long < minLong:
        minLong = apr_14_trips[i].long
        
    if location_compare(apr_14_trips[i].long, apr_14_trips[i].lat, -73.9875, 40.75, .04, .075) == 1:
        conc_trips.append(Trip(new_york_apr_14[i]))


        
    
        
 



#%% LOCATION HISTOGRAM %%#
division = 500
location_histogram, long_edges, lat_edges = np.histogram2d([(trips.long) for trips in apr_14_trips],[(trips.lat) for trips in apr_14_trips], bins = [int((maxLong-minLong)*division),int((maxLat-minLat)*division)], range = [[minLong, maxLong],[minLat, maxLat]])

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

loc_hist_fin, long_edges_final, lat_edges_final = np.histogram2d([(trips.long) for trips in conc_trips],[(trips.lat) for trips in conc_trips], bins = [int((0.08)*division),int((0.15)*division)], range = [[-74.0275, -73.9475],[40.675,40.825]])
loc_hist_fin = loc_hist_fin.T

#%% PLOT %%#
fig, ax = plt.subplots(figsize=(8,5))
X_loc,Y_loc = np.meshgrid(long_edges_final,lat_edges_final)
plt.pcolormesh(X_loc,Y_loc,loc_hist_fin,cmap = cm.cool)
plt.colorbar()


        


        
        
        

