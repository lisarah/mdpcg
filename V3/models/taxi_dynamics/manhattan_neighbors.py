# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 12:09:47 2021

Kernel of Manhattan's MDP dynamics.

103, 104, and 105 are individual island zones, they will be removed from the 
MDP model.

@author: Sarah Li, Nico Miguel
"""
zone_neighbors = {
	4: [79, 224, 232],
	12: [13, 88, 261, 999],
	13: [12, 231, 261],
	24: [41, 43, 151, 166],
	41: [24, 42, 43, 74, 75, 166],
	42: [41, 74, 116, 120, 152, 999],
	43: [24, 41, 75, 151, 236, 238],
	45: [144, 148, 209, 231, 232, 999],
	48: [50, 68, 100, 142, 163, 230],
	50: [48, 143, 246],
	68: [48, 90, 100, 158, 186, 246, 249],
	74: [41, 42, 75], #, 194, 999
	75: [41, 43, 74, 236, 262, 263],
	79: [4, 107, 113, 114, 148, 224],
	87: [88, 209, 261],
	88: [12, 87, 261],
	90: [68, 186, 234, 249],
	100: [48, 68, 164, 186, 230],
	107: [79, 137, 170, 224, 234],
	113: [79, 114, 234, 249],
	114: [79, 113, 125, 144, 211, 249],
	116: [42, 152, 244],
	120: [42, 127, 243, 244, 999],
	125: [114, 158, 211, 231, 249, 999],
	127: [120, 128, 243],
	128: [127, 243], #, 153, 999
	137: [107, 170, 224, 233],
	140: [141, 229, 262], #202, 
	141: [140, 229, 237, 263],
	142: [43, 48, 143, 163, 239],
	143: [50, 142, 239],
	144: [45, 114, 211, 231],
	148: [45, 79, 144, 232],
	151: [24, 43, 238],
	152: [42, 116, 166],
    # 153: [128, 999],
	158: [68, 125, 246, 249],
	161: [162, 163, 164, 170, 230],
	162: [161, 163, 170, 229, 233, 237],
	163: [43, 48, 142, 161, 162, 230, 237],
	164: [100, 161, 170, 186, 234],
	166: [24, 41, 152],
	170: [107, 137, 161, 162, 164, 233],
	186: [68, 90, 100, 164, 234],
    # 194: [74, 999],
    # 202: [140, 999],
    209: [45, 87, 231, 261],
	211: [114, 125, 144, 231],
	224: [4, 79, 107, 137],
	229: [140, 141, 162, 233],
	230: [48, 100, 161, 163],
	231: [13, 45, 125, 144, 209, 211, 261],
	232: [4, 45, 148, 999],
	233: [137, 162, 170, 229, 999],
	234: [90, 107, 113, 164, 186],
	236: [43, 75, 141, 237, 263],
	237: [43, 141, 161, 162, 236],
	238: [43, 151, 239],
	239: [43, 142, 143, 238],
	243: [120, 127, 128, 244, 999],
	244: [116, 120, 243],
	246: [48, 50, 68, 158, 999],
	249: [68, 90, 113, 114, 125, 158],
	261: [12, 13, 87, 88, 209, 231],
    262: [75, 140, 263],
	263: [75, 141, 236, 262],
    # 999: [42, 74, 194, 202, 120, 153, 128, 233, 243, 45, 12, 125, 246],
}
# Excluding state 999 for now
exclude_states = [153, 194, 202, 999]
for zone, neighbors in zone_neighbors.items():
    for state in exclude_states:
        if state in neighbors:
            neighbors.remove(state)


def zone_to_state(neighbors_dict):
    """ Converts a zone based dictionary to [0,S] index-based indices.
    
    Args:
        neighbors_dict: a zone-based dictionary, where the key is the zone 
            number, and the entry is a list of neighboring zones.
    Returns:
        new_keys: a dictionary with [zone_ind] -> [state_ind] as items. 
    """
    new_keys = {}
    s_ind = 0
    for state in neighbors_dict.keys():
        # if state in [153, 194, 202]: 
        #     # s_ind += 1
        #     continue
        new_keys[state] = s_ind
        s_ind +=1
    return new_keys


def state_neighbors_dictionary(neighbors_dict):
    """ Unify neighbors dict to have 0 to S state indices.
    
    Args:
        neighbors_dict: a dictionary of neighbors with non-uniform indices.
    Returns:
        new_dict: a dictionary of the same neighbors with uniform indices.
    """         
    state_ind  = zone_to_state(neighbors_dict)
    state_dictionary = {}
    for origin_zone, zone_neighbors in neighbors_dict.items():
        state_neighbors = []
        for zone in zone_neighbors:
            state_neighbors.append(state_ind[zone])
        state_dictionary[state_ind[origin_zone]] = state_neighbors
    return state_dictionary


STATE_NEIGHBORS = state_neighbors_dictionary(zone_neighbors)


def most_neighbors(neighbors_dict):
    max_neighbors = 0
    for state, neighbors in neighbors_dict.items():
        max_neighbors = max([len(neighbors), max_neighbors])
    return max_neighbors

ZONE_IND = {
	'Alphabet City': 4,
	'Battery Park': 12,
	'Battery Park City': 13,
	'Bloomingdale': 24,
	'Central Harlem': 41,
	'Central Harlem North': 42,
	'Central Park': 43,
	'Chinatown': 45,
	'Clinton East': 48,
	'Clinton West': 50,
	'East Chelsea': 68,
	'East Harlem North': 74,
	'East Harlem South': 75,
	'East Village': 79,
	'Financial District North': 87,
	'Financial District South': 88,
	'Flatiron': 90,
	'Garment District': 100,
# 	"Governor's Island": 103,
# 	"Ellis Island": 104,
# 	"Liberty Island": 105,
	'Gramercy': 107,
	'Greenwich Village North': 113,
	'Greenwich Village South': 114,
	'Hamilton Heights': 116,
	'Highbridge Park': 120,
	'Hudson Sq': 125,
	'Inwood': 127,
	'Inwood Hill Park': 128,
	'Kips Bay': 137,
	'Lenox Hill East': 140,
	'Lenox Hill West': 141,
	'Lincoln Square East': 142,
	'Lincoln Square West': 143,
	'Little Italy/NoLiTa': 144,
	'Lower East Side': 148,
	'Manhattan Valley': 151,
	'Manhattanville': 152,
	'Marble Hill': 153,
	'Meatpacking/West Village West': 158,
	'Midtown Center': 161,
	'Midtown East': 162,
	'Midtown North': 163,
	'Midtown South': 164,
	'Morningside Heights': 166,
	'Murray Hill': 170,
	'Penn Station/Madison Sq West': 186,
	'Randalls Island': 194,
	'Roosevelt Island': 202,
	'Seaport': 209,
	'SoHo': 211,
	'Stuy Town/Peter Cooper Village': 224,
	'Sutton Place/Turtle Bay North': 229,
	'Times Sq/Theatre District': 230,
	'TriBeCa/Civic Center': 231,
	'Two Bridges/Seward Park': 232,
	'UN/Turtle Bay South': 233,
	'Union Sq': 234,
	'Upper East Side North': 236,
	'Upper East Side South': 237,
	'Upper West Side North': 238,
	'Upper West Side South': 239,
	'Washington Heights North': 243,
	'Washington Heights South': 244,
	'West Chelsea/Hudson Yards': 246,
	'West Village': 249,
	'World Trade Center': 261,
	'Yorkville East': 262,
	'Yorkville West': 263
}
    
    
    