# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:25:17 2021

Process New York City Uber driver data.

@author: Nico Miguel, Sarah Li

Trips in this file are from New York, December 2019. 2020 data was avoided for now due to the global pandemic disrupting normal trip operations.

Analysis was carried out using trips in Manhattan

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
import numpy as np
import matplotlib.pyplot as plt
import util.trip as trip
import models.taxi_dynamics.visualization as visual
from haversine import haversine
import pickle
#%% DATA %%#
# Import trip data into numpy array
months = ['dec', 'jan']
partition_amount = [12, 15] # 12 = 15 min intervals vs 15 = 12  min intervals

# Import taxi zone lookup table
taxi_zone_file = "taxi+_zone_lookup.csv"
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

area = 'Manhattan'
Man_zones = zone_list(area)
Man_zones.remove(103)
Man_zones.remove(104)
Man_zones.remove(105)
Man_zones.remove(153)
Man_zones.remove(194)
Man_zones.remove(202)
Man_zones = np.asarray(Man_zones)


#%% TAU CALCULATION %%#

def tau_calculation(borough_time_sorted_trips):
    tau = np.mean(np.array([trips.trip_time for trips in borough_time_sorted_trips]))
    
    return tau

#%% ZONE HISTOGRAM %%#
# Define trip occurence plotting function
# Input: list of trips beginning in borough of interest | array of zones in borough of interest | string containing borough name
# Output: Array and plot of trip occurrences per borough zone 
def trip_plot(borough_trips, zone_list, borough_name, plot=False):
    hist = np.histogram([t.zone_pu for t in borough_trips], bins = zone_list)

    if plot:    
        zone_names = [str(x) for x in np.delete(zone_list, -1)]
    
        fig = plt.figure(figsize=(20,5))
        ax = fig.add_axes([0,0,1,1])
        ax.bar(zone_names, hist[0])
    
        title = 'Trips Originating in ' + borough_name + ', January 2019'
        ax.set_title(title)
        ax.set_xlabel('Zone Number')
        ax.set_ylabel('Trip Occurrences')
        plt.show()
    
    return hist

# #%% STATE TRANSITION PROBABILITY %%#
# # Define transition probability matrix function
# # For "n" number of zones in a borough, function creates matrix size (n+1 x n+1) to represent transition probabilities.
# # Rows represent the zone that trip originated in (pick up) and column represents the zone that trip finished at (drop off).
# # Final column and row are for "external" borough--not all trips stayed in the same overall borough. 
# # Hence, final column represents the probability that a trip starting in a zone of interest finished in another borough. 
# # Final row is only zeros for now, because no trips analyzed begin outside of borough of interest. This will be fixed later on
# # to accomodate trips that start outside borough of interest and end in the borough of interest. Additionally, "external" borough column/row
# # will be modified to detail exactly which borough the trips start or end in, once multi-borough analysis is carried out. 
# #
# # Input: list of trips beginning in borough of interest | trip occurrence data from histogram/plot function
# # Output: (n) x (n) matrix of transition probabilities. Probability of trip starting in enumerated zone i and ending in enumerated zone j 
# # is given by value at (i,j,0) | (n) x (n) matrix of origin-destination trip counts. Number of trips starting in enumerated zone i and ending 
# # in enumerated zone j is given by value at (i,j,1)


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
    
    return [state_matrix, count_matrix]
     
# # Create transition matrix      
# #StIsl_matrix = state_transition_matrix(StIsl_trips, StIsl_trip_hist)           
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
#%% compute distance %%#
compute_distance = True
if compute_distance:
    print('compute distance matrices')
    distance_matrix = np.zeros([len(Man_zones)-1,len(Man_zones)-1])
    
    # # Distance Matrix
    for i in range(len(Man_zones)-1):
        for j in range(len(Man_zones)-1):
            if i == j:
                distance_matrix[i,j] = 1.609 # trips occurring in only one zone are 6 miles
            else:
                zone_i_latlon = visual.get_zone_locations('Manhattan')[Man_zones[i]]
                zone_j_latlon = visual.get_zone_locations('Manhattan')[Man_zones[j]]
                distance_matrix[i,j] = haversine(zone_i_latlon, zone_j_latlon)
    np.savetxt('distance_matrix.csv', distance_matrix, delimiter=',')
     
#%% TRIPS %%#
# Create list of trip objects
for month in months:
    month_int = '12' if month == 'dec' else '01'
    data_filename = f"yellow_tripdata_2019-{month_int}.csv" 
    
    print(f' opening file {data_filename}')
    new_york_2019 = pd.read_csv(data_filename, header=0).to_numpy()
    trips = []
    for i in range(new_york_2019.shape[0]):
        print(f'\r creating trip objects {i}/{new_york_2019.shape[0]}     ', end='')
        trips.append(trip.Trip(new_york_2019[i]))
    print('')
    print('getting out boroughs: ')
    manhattan_trips = [t for t in trips 
                       if t.zone_pu in Man_zones and t.zone_do in Man_zones \
                           and t.putime > 9 and t.putime < 12]
    # Save trips into a pickle
    output_filename = f'models/taxi_data/manhattan_trips_{month}.pickle'
    print(f'saving trip list {output_filename} ')    
    open_file = open(output_filename, 'wb')
    pickle.dump(manhattan_trips, open_file)
    open_file.close()    
    # some extra code to do plotting/average compuations
    # Man_tau = tau_calculation(Man_trips_rush_hour)   
    # Man_trip_hist = trip_plot(Man_trips_rush_hour, Man_zones, area)   
    # Man_transition_matrix = state_transition_matrix(Man_trips_rush_hour, Man_trip_hist)
    for p_min in partition_amount:
        t_min = 15 if p_min == 12 else 12
        print(f' running for the month of {month} for {t_min} min intervals.')
        count_csv_filename = f'models/taxi_data/count_kernel_{month}_{t_min}min.csv'
        transition_csv_filename = f'models/taxi_data/transition_kernel_{month}_{t_min}min.csv'
        avg_filename = f'models/taxi_data/weighted_average_{month}_{t_min}min.csv'
        #%% Partitioned State Transition and Trip Matrices %%#
        print('creating time partitioned transitions')
        time_partition_bins = np.linspace(9,12,t_min+1)
        
        # # Create histogram for number of trips per 12 minute partition
        partitioned_hist = np.histogram([
            t.putime for t in manhattan_trips], bins=time_partition_bins)
        # List of lists containing trips corresponding to each partition
        rush_hour_trips_parted = partition(manhattan_trips, partitioned_hist) 
    
        # List of state matrices
        Man_partitioned_transitions = [[] for _ in rush_hour_trips_parted]
        
        for count,_ in enumerate(Man_partitioned_transitions):
            part_hist = trip_plot(rush_hour_trips_parted[count], 
                                  Man_zones, area)
            Man_partitioned_transitions[count] = state_transition_matrix(
                rush_hour_trips_parted[count], part_hist)
        # state transition kernel for each timestep    
        print(f'saving transition file {transition_csv_filename}')
        transition_kernel = np.array(
            [matrix[0] for matrix in Man_partitioned_transitions]) 
        transition_df = pd.DataFrame(np.hstack(transition_kernel))
        transition_df.to_csv(transition_csv_filename, index = False)
        
        # state trip count matrix for each timestep
        print(f'saving count kernel file {count_csv_filename}')
        count_kernel = np.array([mat[1] for mat in Man_partitioned_transitions]) 
        count_df = pd.DataFrame(np.hstack(count_kernel))
        count_df.to_csv(count_csv_filename, index = False)
    
        #%% Weighted Average Trip Distance Per Origin State %%#
        weighted_average_distance = [[] for _ in rush_hour_trips_parted]
        
        for count,_ in enumerate(rush_hour_trips_parted):
            weights = Man_partitioned_transitions[count][0]
            array_ = []
            for i in range(len(Man_zones)-1):
                array_.append(np.matmul(weights[i,:], distance_matrix[:,i]))
                    
            weighted_average_distance[count] = array_
        print(f'saving average weight file {avg_filename}')
        np.savetxt(avg_filename, weighted_average_distance, delimiter=',')
    
    



