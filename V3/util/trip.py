# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:26:37 2021

Extract a trip from a CSV row.

@author: Nico Miguel, Sarah Li
"""
import datetime

#%% memes %%#

import models.taxi_dynamics.visualization as visual


Man_zones = [4,  12,  13,  24,  41,  42,  43,  45,  48,  50,  68,  74,  75,
        79,  87,  88,  90, 100, 103, 104, 105, 107, 113, 114, 116, 120,
       125, 127, 128, 137, 140, 141, 142, 143, 144, 148, 151, 152, 153,
       158, 161, 162, 163, 164, 166, 170, 186, 194, 202, 209, 211, 224,
       229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246,
       249, 261, 262, 263]

# Define compare function
def location_compare(trip_long, trip_lat, area_long, area_lat, bound_long, 
                     bound_lat):
    long_check = abs(trip_long - area_long) <= bound_long
    lat_check = abs(trip_lat - area_lat) <= bound_lat

    return long_check and lat_check
    
# Define trip class
class Trip:
    def __init__(self, trip_instance):
        # in the newer CSV version,there's an additional first entry
        trip_instance = trip_instance[1:]
        # Pickup Date retrieval
        pudate_ = trip_instance[1].split(" ")[0]
        self.pudate = datetime.datetime.strptime(pudate_, '%Y-%m-%d')  #'%m/%d/%Y'
        
        # Dropoff Date retrieval
        dodate_ = trip_instance[2].split(" ")[0]
        self.dodate = datetime.datetime.strptime(dodate_, '%Y-%m-%d')  #'%m/%d/%Y'
        
        # Pickup Time retrieval
        putime_ = trip_instance[1].split(" ")[1]
        putime = putime_.split(":")
        pu_hours_ = int(putime[0])
        pu_minutes_ = int(putime[1])
    
        elapsed_pu = datetime.timedelta(hours = pu_hours_, minutes = pu_minutes_, 
                                     seconds = 0)
        self.putime = elapsed_pu.seconds/3600 # Time in hours since beginning of day
                
        # Dropoff Time retrieval
        dotime_ = trip_instance[2].split(" ")[1]
        dotime = dotime_.split(":")
        do_hours_ = int(dotime[0])
        do_minutes_ = int(dotime[1])
        
        elapsed_do = datetime.timedelta(hours = do_hours_, minutes = do_minutes_, 
                                     seconds = 0)
        self.dotime = elapsed_do.seconds/3600 # Time in hours since beginning of day
        
        # Elapsed Trip Time
        self.trip_time = self.dotime - self.putime
        
        # Pickup Location retrieval
        self.zone_pu = trip_instance[7]
        # if self.zone_pu in Man_zones:
        #     self.pu_latlon = visual.get_zone_locations('Manhattan')[self.zone_pu]
        
        
        # Dropoff Location retrieval
        self.zone_do = trip_instance[8]
        # if self.zone_do in Man_zones:
        #     self.do_latlon = visual.get_zone_locations('Manhattan')[self.zone_do]
        
        
        # Cost of ride
        self.fare = trip_instance[10]
        
