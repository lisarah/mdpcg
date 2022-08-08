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
# import util.trip as manhattan
import models.taxi_dynamics.manhattan_neighbors as m_neighs

""" Format matplotlib output to be latex compatible with gigantic fonts."""
# mpl.rc('font',**{'family':'serif'})
# mpl.rc('text', usetex=True)
# mpl.rcParams.update({'font.size': 12})
# mpl.rc('legend', fontsize='small')


directory = 'C:/Users/Sarah Li/Desktop/code/mdpcg/V3/models/taxi_data/' 
months = ['dec', 'dec', 'jan', 'jan']
t_ints = [12, 15, 12, 15]

for month, t_int in zip(months, t_ints):
    t_int_hr = t_int/60
    trips_filename = directory+f'manhattan_trips_{month}.pickle'
    output_filename = directory+f'manhattan_transitions_{month}_{t_int}min.pickle'
    print(f'opening file {trips_filename}')
    trips_file = open(trips_filename, 'rb')
    trips = pickle.load(trips_file)
    trips_file.close()
    
    # get rid of all trips with time that are negative or too long
    print('removing negative trip times and >90 trip times.')
    trip_times = [t.trip_time * 60 for t in trips]
    avg_time = sum(trip_times) / len(trip_times) 
    print(f' average trip time is {avg_time} ') # 13.28 for manhattan
    # process based on trip times:
    # exclude longer than 1.5 hours
    large_rides =[i for i, x in enumerate(trip_times) if x  > 90] 
    # exclude negative times
    negative_rides = [i for i, x in enumerate(trip_times) if x < 0]
    
    processed_trips = [trips[i] for i in range(len(trips)) 
                       if i not in large_rides+negative_rides]
    processed_trip_times = [t.trip_time * 60 for t in processed_trips]
    avg_time = sum(processed_trip_times) / len(processed_trip_times) 
    print(f' average processed trip time is {avg_time}') # 13.45 for processed manhattan
    
    # plot a histogram of the travelling times
    plt.figure()
    sea.histplot(processed_trip_times, binwidth=t_int) # , kde=True
    plt.yscale('log')
    plt.xlabel('Trip Time (min)')
    plt.ylabel('Trip Occurrences')
    plt.title(f'{month} with {t_int} min intervals')
    plt.show()
    
    # define all states: 
    zone_list = list(m_neighs.zone_neighbors.keys())
    if t_int == 12: 
        total_time = 15
    elif t_int == 15:
        total_time = 12
    else:
        assert t_int in [12,15], 'give new time interval'
    time_bins = np.linspace(9, 12, total_time+1) # 15 min intervals
    
    # sort trips by time    
    processed_trips.sort(key=lambda x: x.putime) # sort trips by time
    time_freqs, _ = np.histogram([t.putime for t in processed_trips], 
                                      bins = time_bins)    
    transitions = []
    last_t_ind = 0
    max_q = 7 if t_int == 12 else 6
    assert t_int in [12, 15], 'give new maximum queue level'
    
    for t in range(total_time):
        transitions.append({z: {} for z in zone_list})
        print(f'\r on trip {last_t_ind}/{sum(time_freqs)}', end  = '  ')
        trips_in_t = processed_trips[last_t_ind:last_t_ind+time_freqs[t]]
        for trip in trips_in_t:
            if trip.zone_pu in zone_list and trip.zone_do in zone_list:
                trip_time = min(int(trip.trip_time/t_int_hr), max_q)
                
                dest_tuple = (trip.zone_do, trip_time)
                if dest_tuple in transitions[-1][trip.zone_pu]: 
                    transitions[-1][trip.zone_pu][dest_tuple] += 1
                else:
                    transitions[-1][trip.zone_pu][dest_tuple] = 1
                
        for pu_zone in zone_list:
            total_trips = sum(list(transitions[-1][pu_zone].values()))
            transitions[-1][pu_zone] = {k: v/total_trips  for k, v 
                                        in transitions[-1][pu_zone].items()}          
        last_t_ind += time_freqs[t]
    print('')
    
    """
    Transitions is a list of dictionaries, where the ith element is the dictionary
    of transitions for timestep i. Each dictionary element has the following format
    key = tuple(zone_ind, trip_time):
            - zone_ind - index of destination zone in TLC's notation.
            - trip time - from 1 - 5, each corresponding to a 15 min increment
    value = probability of arriving in zone_ind with trip_time 
    """
    print(f'saving file {output_filename}')
    open_file = open(output_filename, 'wb')
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