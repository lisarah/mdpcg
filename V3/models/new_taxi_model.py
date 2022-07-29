# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 08:46:43 2022

@author: craba
"""
import pickle 
import numpy as np
import seaborn as sea
import matplotlib as mpl
import matplotlib.pyplot as plt
import util.trip as manhattan
import models.taxi_dynamics.manhattan_neighbors as m_neighs

""" Format matplotlib output to be latex compatible with gigantic fonts."""
mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 12})
mpl.rc('legend', fontsize='small')


directory = 'C:/Users/craba/Desktop/code/mdpcg/V3/' 
trips_filename = directory+'models/manhattan_trips_jan.pickle'
trips_file = open(trips_filename, 'rb')
trips = pickle.load(trips_file)
trips_file.close()

zone_dict = {}
for t in trips:
    print(f'\r on trip {trips.index(t)}/{len(trips)}', end  = '  ')
    zone_dict[t.zone_do] = 0
    zone_dict[t.zone_pu] = 0
    if t.zone_do not in m_neighs.zone_neighbors or t.zone_pu not in m_neighs.zone_neighbors:
        trips.remove(t)
zone_list = list(zone_dict.keys())
zone_list.sort()
print(zone_list)

trip_times = [t.trip_time * 60 for t in trips]
avg_time = sum(trip_times) / len(trip_times) 
print(f' average trip time is {avg_time} ') # 13.28 for manhattan
# process based on trip times:
# exclude longer than 2 hours
large_rides =[i for i, x in enumerate(trip_times) if x  > 120] 
# exclude negative times
negative_rides = [i for i, x in enumerate(trip_times) if x < 0]

processed_trips = [trips[i] for i in range(len(trips)) 
                   if i not in large_rides+negative_rides]
processed_trip_times = [t.trip_time * 60 for t in processed_trips]
avg_time = sum(processed_trip_times) / len(processed_trip_times) 
print(f' average processed trip time is {avg_time}') # 13.45 for processed manhattan

# plot a histogram of the travelling times
sea.histplot(processed_trip_times, binwidth=15) # , kde=True
plt.yscale('log')
plt.xlabel('Trip Time (min)')
plt.ylabel('Trip Occurrences')
plt.show()

# define all states: 
zone_bins  = manhattan.Man_zones
total_time = 12
time_bins = np.linspace(9, 12, total_time) # 15 min intervals

# sort trips by time    
processed_trips.sort(key=lambda x: x.putime) # sort trips by time
time_freqs, _ = np.histogram([t.putime for t in processed_trips], 
                                  bins = time_bins)    
transitions = []
last_t_ind = 0
for t in range(total_time-1):
    transitions.append({z: {} for z in zone_bins})
    # print(last_t_ind+time_freqs[t])
    trips_in_t = processed_trips[last_t_ind:last_t_ind+time_freqs[t]]
    for trip in trips_in_t:
        if trip.zone_pu in zone_bins:
            trip_time = min(int(trip.trip_time/0.25), 4) +1
            
            dest_tuple = (trip.zone_do, trip_time)
            # if trip_time !=  1:
            #     print(f'{dest_tuple} from {trip.zone_pu} has greater transition')

                
            if dest_tuple in transitions[-1][trip.zone_pu]: 
                transitions[-1][trip.zone_pu][dest_tuple] += 1
            else:
                transitions[-1][trip.zone_pu][dest_tuple] = 1
            
    for pu_ind in zone_bins:
        total_trips = sum(list(transitions[-1][pu_ind].values()))
        for key in transitions[-1][pu_ind].keys():
            transitions[-1][pu_ind][key] = transitions[-1][pu_ind][key] / total_trips
                
    last_t_ind += time_freqs[t]


"""
Transitions is a list of dictionaries, where the ith element is the dictionary
of transitions for timestep i. Each dictionary element has the following format
key = tuple(zone_ind, trip_time):
        - zone_ind - index of destination zone in TLC's notation.
        - trip time - from 1 - 5, each corresponding to a 15 min increment
value = probability of arriving in zone_ind with trip_time 
"""
open_file = open('models/manhattan_transitions_jan.pickle', 'wb')
pickle.dump(transitions, open_file)
open_file.close()
# processed_destinations = [t.zone_pu for t in processed_trips]
# freq_mat, _, _ = np.histogram2d(processed_destinations, processed_trip_times, 
#                       bins=(zone_bins, time_bins))



# # define transition matrix
# borough_trip_list.sort(key = lambda x: x.zone_pu)

# state_matrix = np.empty([len(zones), len(zones)])
# count_matrix = np.empty([len(zones), len(zones)])
# array_total = []

# index = 0
# for i in range(len(zones)):
#     index2 = index+freq[i]
#     total_ = freq[i]
#     array_total.append(total_)
#     trips_ = borough_trip_list[index:index2]
#     for j in range(len(zones)):
#         if total_ == 0:
#             state_matrix[i,j] = 0
#             count_matrix[i,j] = 0
#         counter = 0
#         for trip_instance in trips_:
#             if trip_instance.zone_do == zones[j]:
#                 counter += 1
#         state_matrix[i,j] = counter/total_    
#         count_matrix[i,j] = counter            
#     index += freq[i]

# np.nan_to_num(state_matrix)
# np.nan_to_num(count_matrix)
# # np.nan_to_num(array_total)
# # ext_state_col = np.zeros((len(zones),1))
# # ext_state_row = np.zeros((len(zones)+1))
# # ext_count_col = np.zeros((len(zones),1))
# # ext_count_row = np.zeros((len(zones)+1))
# # for i in range(len(zones)):
# #     ext_state_col[i] = 1.000000 - np.sum(state_matrix[i,:-1])
# #     ext_count_col[i] = array_total[i] - np.sum(count_matrix[i,:-1])

# # state_matrix = np.hstack((state_matrix, ext_state_col))

# # state_matrix = np.vstack((state_matrix, ext_state_row))

# # count_matrix = np.hstack((count_matrix, ext_count_col))

# # count_matrix = np.vstack((count_matrix, ext_count_row))


# return [state_matrix, count_matrix]

# manhattan = get_state_transition(hist_mat, )