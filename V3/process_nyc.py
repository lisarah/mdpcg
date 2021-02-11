# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:25:17 2021

Process New York City Uber driver data.

@author: Nico Miguel, Sarah Li

Trips in this file are from New York, January 2019. 2020 data was avoided for now due to the global pandemic disrupting normal trip operations.

Example analysis was carried out using trips originating in Staten Island. borough

USAGE:
    There are four functions that drive the data analysis.
    
    Trip locations are defined by boroughs and zones, instead of specific latitude/longitude data.
    In New York, there are five boroughs: the Bronx, Brooklyn, Manhattan, Queens, and Staten Island. Each borough has anywhere from 40 to 60 zones 
    that encompass different neighborhoods and general areas inside the borough. The zones are numbered, an overall range from 1 to 265.
    
    The zone lookup file matches zones to their boroughs, since they aren't numbered sequentially. 
    
    Trip data is imported into a pandas DataFrame. The trips are passed as objects in order to easily access their specific metadata.  
    
    First, the zone_list function creates an array of the zone numbers in the borough of interest. Then, the borough_trips function parses 
    through the entire list of trips and extracts the trips that originate in any of the zones that are in your borough of interest.
    Trip_plot creates both a plot and an array of the trip occurrences. The plot displays the 'histogram' data, showing the frequency 
    of trips for each zone in the borough. The array output is simply the histogram/bar plot in numerical form, with the first array tracking
    the frequency of trips in zone order and the second array keeping track of the zones. 
    Finally, the state_transition_matrix function iterates through each trip and keeps track of their destination, in order to create a 
    prior probability matrix to use as a starting point for the MDP.
    
    Next steps are to create an additional matrix that averages the trip time for each origin-destination pair present in the transition probability matrix.
    
    
    
"""
import pandas as pd
import decimal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import math
import util.trip as trip
from collections import defaultdict
import models.taxi_dynamics.visualization as visual
from haversine import haversine
import csv
#%% DATA %%#
# Import trip data into numpy array
data_file_name = "D:\\Behcet\\GameTheory\\NewYorkUber\\yellow_tripdata_2019-01.csv"
new_york_jan_2019 = pd.read_csv(data_file_name, header = 0).to_numpy()

# Import taxi zone lookup table
taxi_zone_file = "D:\\Behcet\\GameTheory\\NewYorkUber\\taxi+_zone_lookup.csv"
taxi_zone_lookup = pd.read_csv(taxi_zone_file, header = 0)

#%% BOROUGH ZONES %%#
# Define borough zone lookup function 
# Input: string with borough name, options are "Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island" 
# Output: numpy array of zones that are in chosen borough
def zone_list(borough_name):
    borough_zone_df = taxi_zone_lookup.loc[taxi_zone_lookup['Borough'] == borough_name]
    borough_zone_array = borough_zone_df['LocationID'].to_numpy()
    borough_zone_array = np.append(borough_zone_array, borough_zone_array[-1])
    
    return borough_zone_array.tolist()
    
    

#%% TRIPS %%#
# Create list of trip objects
nyjan2019_trips = []


for i in range(new_york_jan_2019.shape[0]):
    nyjan2019_trips.append(trip.Trip(new_york_jan_2019[i]))

#%% TRIP SORTING %%#
# Define borough trip sorting function
# Input: array of borough zones | list of overall trips to be sorted 
# Output: list of trips that begin in borough of interest
def borough_trips(trip_list, borough_zones_list):
    borough_trip_list = []
    for count, trip_instance in enumerate(trip_list):
        if trip_instance.zone_pu and trip_instance.zone_do in borough_zones_list:
            borough_trip_list.append(trip_list[count])
    
    return borough_trip_list

# Define time sorting function
# Input: list of trips that begin in borough of interest | time bounds (decimal)
# Output: list of trips in borough that begin within certain time-range
def time_sorted_trips(borough_trip_list, timeA, timeB):
    boolean_index = np.array([False for trip in borough_trip_list])
    for count, trip_instance in enumerate(borough_trip_list):
        if trip_instance.putime > 9 and trip_instance.putime < 12:
            boolean_index[count] = True
    
    borough_trip_list_np = np.array(borough_trip_list)
    time_sorted_trip = borough_trip_list_np[boolean_index]
    
    borough_time_sorted_trips = time_sorted_trip.tolist()
    
    return borough_time_sorted_trips


# Extract all trips originating in Staten Island Borough
# area = 'Staten Island'
# StIsl_zones = zone_list(area)
# StIsl_trips = borough_trips(nyjan2019_trips, StIsl_zones)

area = 'Manhattan'
Man_zones = zone_list(area)
Man_zones.remove(104)
Man_zones.remove(105)
Man_zones = np.asarray(Man_zones)
Man_trips = borough_trips(nyjan2019_trips, Man_zones)
Man_trips_rush_hour = time_sorted_trips(Man_trips, 9, 12)

#%% TAU CALCULATION %%#

def tau_calculation(borough_time_sorted_trips):
    tau = np.mean(np.array([trips.trip_time for trips in borough_time_sorted_trips]))
    
    return tau
    
Man_tau = tau_calculation(Man_trips_rush_hour)


#%% ZONE HISTOGRAM %%#
# Define trip occurence plotting function
# Input: list of trips beginning in borough of interest | array of zones in borough of interest | string containing borough name
# Output: Array and plot of trip occurrences per borough zone 
def trip_plot(borough_trips, borough_zones_list, borough_name):
    hist = np.histogram([trips.zone_pu for trips in borough_trips], bins = borough_zones_list)    
    borough_zones_str = [str(x) for x in np.delete(borough_zones_list, -1)]

    fig = plt.figure(figsize=(20,5))
    ax = fig.add_axes([0,0,1,1])
    ax.bar(borough_zones_str, hist[0])

    title = 'Trips Originating in ' + borough_name + ', January 2019'
    ax.set_title(title)
    ax.set_xlabel('Zone Number')
    ax.set_ylabel('Trip Occurrences')
    plt.show()
    
    return hist
    
# Count and sort all trips originating in Staten Island
#StIsl_trip_hist = trip_plot(StIsl_trips, StIsl_zones, area)

Man_trip_hist = trip_plot(Man_trips_rush_hour, Man_zones, area)


#%% STATE TRANSITION PROBABILITY %%#
# Define transition probability matrix function
# For "n" number of zones in a borough, function creates matrix size (n+1 x n+1) to represent transition probabilities.
# Rows represent the zone that trip originated in (pick up) and column represents the zone that trip finished at (drop off).
# Final column and row are for "external" borough--not all trips stayed in the same overall borough. 
# Hence, final column represents the probability that a trip starting in a zone of interest finished in another borough. 
# Final row is only zeros for now, because no trips analyzed begin outside of borough of interest. This will be fixed later on
# to accomodate trips that start outside borough of interest and end in the borough of interest. Additionally, "external" borough column/row
# will be modified to detail exactly which borough the trips start or end in, once multi-borough analysis is carried out. 
#
# Input: list of trips beginning in borough of interest | trip occurrence data from histogram/plot function
# Output: (n) x (n) matrix of transition probabilities. Probability of trip starting in enumerated zone i and ending in enumerated zone j 
# is given by value at (i,j,0) | (n) x (n) matrix of origin-destination trip counts. Number of trips starting in enumerated zone i and ending 
# in enumerated zone j is given by value at (i,j,1)


def state_transition_matrix(borough_trip_list, borough_hist):
    freq = borough_hist[0]
    zones = np.delete(borough_hist[1], -1)
    
    borough_trip_list.sort(key = lambda x: x.zone_pu)
    
    state_matrix = np.empty([len(zones), len(zones)])
    count_matrix = np.empty([len(zones), len(zones)])
    array_total = []
    
    index = 0
    for i in range(len(zones)):
        index2 = index+freq[i]
        total_ = freq[i]
        array_total.append(total_)
        trips_ = borough_trip_list[index:index2]
        for j in range(len(zones)):
            if total_ == 0:
                state_matrix[i,j] = 0
                count_matrix[i,j] = 0
            counter = 0
            for trip_instance in trips_:
                if trip_instance.zone_do == zones[j]:
                    counter += 1
            state_matrix[i,j] = counter/total_    
            count_matrix[i,j] = counter            
        index += freq[i]
    
    np.nan_to_num(state_matrix)
    np.nan_to_num(count_matrix)
    # np.nan_to_num(array_total)
    # ext_state_col = np.zeros((len(zones),1))
    # ext_state_row = np.zeros((len(zones)+1))
    # ext_count_col = np.zeros((len(zones),1))
    # ext_count_row = np.zeros((len(zones)+1))
    # for i in range(len(zones)):
    #     ext_state_col[i] = 1.000000 - np.sum(state_matrix[i,:-1])
    #     ext_count_col[i] = array_total[i] - np.sum(count_matrix[i,:-1])
    
    # state_matrix = np.hstack((state_matrix, ext_state_col))
    
    # state_matrix = np.vstack((state_matrix, ext_state_row))
    
    # count_matrix = np.hstack((count_matrix, ext_count_col))
    
    # count_matrix = np.vstack((count_matrix, ext_count_row))

    
    return [state_matrix, count_matrix]
     
# Create transition matrix      
#StIsl_matrix = state_transition_matrix(StIsl_trips, StIsl_trip_hist)           
    

Man_transition_matrix = state_transition_matrix(Man_trips_rush_hour, Man_trip_hist)
    
#%% Partitioning trips %%#

# Sort trips into periods of time tau minutes long
timeRange = 3
partition_amount = math.ceil(timeRange/Man_tau)
time_partition_bins = np.linspace(9,12,partition_amount+1)

# Create histogram for number of trips per 12 minute partition
Man_time_partition_hist = np.histogram([trip.putime for trip in Man_trips_rush_hour], bins = time_partition_bins)

def partition(borough_trip_list, partition_hist):
    freq = partition_hist[0]
    time_zones = partition_hist[1]
    borough_trip_list.sort(key = lambda x: x.putime)
    
    trip_partitions = [[] for i in range(len(time_zones)-1)]
    index = 0
    for count,_ in enumerate(trip_partitions):
        index2 = index + freq[count]
        trip_partitions[count] = borough_trip_list[index:index2]
        index += freq[count]

    return trip_partitions

# List of lists containing trips corresponding to each partition
Man_rush_hour_partitioned = partition(Man_trips_rush_hour, Man_time_partition_hist) 

#%% Partitioned State Transition and Trip Matrices %%#

# List of state matrices
Man_partitioned_transitions = [[] for i in range(len(Man_rush_hour_partitioned))]

for count,_ in enumerate(Man_partitioned_transitions):
    part_hist = trip_plot(Man_rush_hour_partitioned[count], Man_zones, area)
    Man_partitioned_transitions[count] = state_transition_matrix(Man_rush_hour_partitioned[count], part_hist)
    

#%% Distance Matrix %%#

distance_matrix = np.zeros([len(Man_zones)-1,len(Man_zones)-1])

# Distance Matrix
for i in range(len(Man_zones)-1):
    for j in range(len(Man_zones)-1):
        if i == j:
            distance_matrix[i,j] = 9.656 # trips occurring in only one zone are 6 miles
        else:
            zone_i_latlon = visual.get_zone_locations('Manhattan')[Man_zones[i]]
            zone_j_latlon = visual.get_zone_locations('Manhattan')[Man_zones[j]]
            distance_matrix[i,j] = haversine(zone_i_latlon, zone_j_latlon)
            
#%% Weighted Average Trip Distance Per Origin State %%#

weighted_average_distance = [[] for i in range(len(Man_rush_hour_partitioned))]

for count,_ in enumerate(Man_rush_hour_partitioned):
    weights = Man_partitioned_transitions[count][0]
    
    array_ = []
    for i in range(len(Man_zones)-1):
        array_.append(np.matmul(weights[i,:], distance_matrix[:,i]))
            
    weighted_average_distance[count] = array_
    
#%% Write to CSV file %%#

np.savetxt('distance_matrix.csv', distance_matrix, delimiter=',')
np.savetxt('weighted_average.csv', weighted_average_distance, delimiter=',')

transition_kernel = np.array([matrix[0] for matrix in Man_partitioned_transitions]) # state transition kernel for each timestep
count_kernel = np.array([matrix[1] for matrix in Man_partitioned_transitions]) # state trip count matrix for each timestep

transition_df = pd.DataFrame(np.hstack(transition_kernel))
count_df = pd.DataFrame(np.hstack(count_kernel))

transition_df.to_csv('transition_kernel.csv', index = False)
count_df.to_csv('count_kernel.csv', index = False)



