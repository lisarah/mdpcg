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
        # Date retrieval
        date_ = trip_instance[0].split(" ")[0]
        date = date_.replace(date_[-4:], date_[-2:])
        self.date = datetime.datetime.strptime(date, '%m/%d/%y')
        
        # Time retrieval
        time_ = trip_instance[0].split(" ")[1]
        time = time_.split(":")
        hours_ = int(time[0])
        minutes_ = int(time[1])
        seconds_ = int(time[2])
        elapsed = datetime.timedelta(hours = hours_, minutes = minutes_, 
                                     seconds = seconds_)
        self.time = elapsed.seconds/3600 # Time in hours since beginning of day
        
        # Pickup Location retrieval
        self.long = trip_instance[2]
        self.lat = trip_instance[1]
        
        # Base
        self.base = trip_instance[3]