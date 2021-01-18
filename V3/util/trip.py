# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:26:37 2021

Extract a trip from a CSV row.

@author: Nico Miguel, Sarah Li
"""
import datetime


# Define compare function
def location_compare(trip_long, trip_lat, area_long, area_lat, bound_long, 
                     bound_lat):
    long_check = abs(trip_long - area_long) <= bound_long
    lat_check = abs(trip_lat - area_lat) <= bound_lat

    return long_check and lat_check
    
# Define trip class
class Trip:
    def __init__(self, trip_instance):
        # Pickup Date retrieval
        pudate_ = trip_instance[1].split(" ")[0]
        self.pudate = datetime.datetime.strptime(pudate_, '%Y-%m-%d')
        
        # Dropoff Date retrieval
        dodate_ = trip_instance[2].split(" ")[0]
        self.dodate = datetime.datetime.strptime(dodate_, '%Y-%m-%d')
        
        # Pickup Time retrieval
        putime_ = trip_instance[1].split(" ")[1]
        putime = putime_.split(":")
        pu_hours_ = int(putime[0])
        pu_minutes_ = int(putime[1])
        pu_seconds_ = int(putime[2])
        elapsed_pu = datetime.timedelta(hours = pu_hours_, minutes = pu_minutes_, 
                                     seconds = pu_seconds_)
        self.putime = elapsed_pu.seconds/3600 # Time in hours since beginning of day
                
        # Dropoff Time retrieval
        dotime_ = trip_instance[2].split(" ")[1]
        dotime = dotime_.split(":")
        do_hours_ = int(dotime[0])
        do_minutes_ = int(dotime[1])
        do_seconds_ = int(dotime[2])
        elapsed_do = datetime.timedelta(hours = do_hours_, minutes = do_minutes_, 
                                     seconds = do_seconds_)
        self.dotime = elapsed_do.seconds/3600 # Time in hours since beginning of day
        
        # Elapsed Trip Time
        self.trip_time = self.dotime - self.putime
        
        # Pickup Location retrieval
        self.zone_pu = trip_instance[7]
        
        
        # Dropoff Location retrieval
        self.zone_do = trip_instance[8]
        
        
        # Cost of ride
        self.fare = trip_instance[-1]
        
        