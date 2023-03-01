"""
Points to Grid

This is a script that map points on a map to grids with specific sizes. 
The grids are flexible i.e. the grids are not adjacent to each other.

Assumptions:
- Measurements are in kilometres (km)
- Grid are squares i.e. a grid size of 0.025 denotes a 25m by 25m base grid size
"""

# Imports
import math
import numpy as np
import pandas as pd
from numba import jit

# Global Constants


class FGP:

    def __init__(self, base_size: float, levels: int = 1, sensitivity: int = 4, equatorialRadius: float = 6378.1370, 
            polarRadius: float = 6356.7523, silent: bool = True):
        
        """
        Main Constructor

        :param base_size: for base grid size e.g. 0.05 for 50m grids
        :param levels: for number of levels to be built up in the grid system e.g. 10
        :param sensitivity: decimal places to round long and lat to
        :param equatorialRadius: float - in km, defaults to 6378.1370
        :param polarRadius: float - in km, defaults to 6356.7523
        """

        if base_size <= 0:
            raise ValueError("base size {} is invalid! Value must be greater than 0".format(base_size))
        
        if levels < 1:
            raise ValueError("levels {} is invalid! Value must be greater than 0".format(levels))

        self.base_size = base_size
        self.levels = levels

        self.equatorial_radius = equatorialRadius
        self.polar_radius = polarRadius
        self.lon_per_deg = (self.equatorial_radius * math.pi) / 180
        self.lat_per_deg = (self.polar_radius * math.pi) / 180

        self.measurement_sensitivity = sensitivity

        # For FGP, deltas are appending to get upper and lower boundaires so deltas as half the real grid size
        self.delta_lat = []
        self.delta_lon = []
        self._getDeltas(silent)

    def printConfig(self):
        """
        Function for view instance parameters and information
        """
        print("-----------------------------------------")
        print("Flexible Point To Grid Configuration")
        print("-----------------------------------------")
        print("Base Grid Size: {0}km by {0}km".format(self.base_size))
        print("Grid Levels: {}".format(self.levels))
        print("Equitorial Radius: {}km".format(self.equatorial_radius))
        print("Polar Radius: {}km".format(self.polar_radius))
        print("Base Latitude Delta: {}".format(self.delta_lat[0]))
        print("Base Longitude Delta: {}".format(self.delta_lon[0]))
        print("Actual Grid Size: {}km by {}km".format(np.round(self.lon_per_deg*self.delta_lon[0], 3), np.round(self.lat_per_deg*self.delta_lat[0], 3)))
        print("-----------------------------------------")

    def generate(self, data: pd.DataFrame, existingGrids: pd.DataFrame):
        locations = data.copy()
        locs_array = locations[['longitude','latitude']].to_numpy()
        loclen = len(locs_array)
        parentIDs = np.full(loclen, -1)
        parent_lon = np.empty(loclen)
        parent_lat = np.empty(loclen)
        parent_lon[:] = np.nan
        parent_lat[:] = np.nan
        
        parents_array = np.array([])
        new_parents = np.array([])
        if existingGrids is not None:
            # parentGrids = self._updateParentBoudaries(existingGrids, 1)
            parents_array = existingGrids[['id', 'centerLon', 'centerLat', 'upperLon', 'lowerLon', 'upperLat', 'lowerLat']].to_numpy()

        for i in range(len(locs_array)):
            # print("---")
            each_lon = locs_array[i][0]
            each_lat = locs_array[i][1]
            which, parentID = self._findParent(each_lon, each_lat, parents_array, new_parents)
            
            if which == "parent":
                parentIDs[i] = parents_array[parentID][0]
                parent_lon[i] = parents_array[parentID][1]
                parent_lat[i] = parents_array[parentID][2]
            elif which == "new":
                parentIDs[i] = -1
                parent_lon[i] = new_parents[parentID][1]
                parent_lat[i] = new_parents[parentID][2]
            else: # totally fresh one
                # create new grid
                upperLon, lowerLon, upperLat, lowerLat = self._getNewGridParams(each_lon, each_lat, 1)
                if len(new_parents) == 0:
                    # new one
                    new_parents = np.array([[0, each_lon, each_lat, upperLon, lowerLon, upperLat, lowerLat]])
                else:
                    # append
                    new_parents = np.concatenate((
                        new_parents,
                        np.array([[0, each_lon, each_lat, upperLon, lowerLon, upperLat, lowerLat]])
                    ), axis = 0)

                parentIDs[i] = -1
                parent_lon[i] = each_lon
                parent_lat[i] = each_lat

        locations['parentID'] = parentIDs
        locations['parent_lon'] = parent_lon
        locations['parent_lat'] = parent_lat
        return locations

    def _getDeltas(self, silent: bool):
        """
        Computes lat and lon deltas
        """
        if not silent: 
            print("Base Delta")
            print('Delta Long without dropping decimal places:',self.base_size / self.lon_per_deg)
            print('Delta Lat without dropping decimal places:',self.base_size / self.lat_per_deg)
        self.delta_lon.append(np.round(((self.base_size / 2) / self.lon_per_deg),self.measurement_sensitivity))
        self.delta_lat.append(np.round(((self.base_size / 2) / self.lat_per_deg),self.measurement_sensitivity))
        if not silent:
            print('Delta Long After dropping decimal places:',self.delta_lon[1])
            print('Delta Lat After dropping decimal places:',self.delta_lat[1])
            print('Reducing decimal point leads to actual grid size (lon):',self.lon_per_deg*self.delta_lon[0])
            print('Reducing decimal point leads to actual grid size (lat):',self.lat_per_deg*self.delta_lat[0])

        for level in range(1, self.levels):
            level_size = (self.base_size / 2) * math.pow(2, level)
            self.delta_lon.append(np.round((level_size / self.lon_per_deg),self.measurement_sensitivity))
            self.delta_lat.append(np.round((level_size / self.lat_per_deg),self.measurement_sensitivity))


    def _getNewGridParams(self, lon, lat, level):
        upperLat = lat + self.delta_lat[level-1]
        lowerLat = lat - self.delta_lat[level-1]
        upperLon = lon + self.delta_lon[level-1]
        lowerLon = lon - self.delta_lon[level-1]
        return upperLon, lowerLon, upperLat, lowerLat

    def _findParent(self, lon, lat, parents_array: np.array, new_parents: np.array):
        # check parents array
        if len(parents_array) > 0:
            parent_match = np.intersect1d(
                    np.intersect1d(
                        np.where(parents_array[:,3] >= lon)[0], 
                        np.where(parents_array[:,4] <= lon)[0]), # lon matches
                    np.intersect1d(
                        np.where(parents_array[:,5] >= lat)[0], 
                        np.where(parents_array[:,6] <= lat)[0]) # lat matches
                ) # both match
            if len(parent_match) > 0:
                return "parent", parent_match[0]
        
        # check new parents
        if len(new_parents) > 0:
            parent_match = np.intersect1d(
                    np.intersect1d(
                        np.where(new_parents[:,3] >= lon)[0], 
                        np.where(new_parents[:,4] <= lon)[0]), # lon matches
                    np.intersect1d(
                        np.where(new_parents[:,5] >= lat)[0], 
                        np.where(new_parents[:,6] <= lat)[0]) # lat matches
                ) # both match
            if len(parent_match) > 0:
                return "new", parent_match[0]
        
        # fresh one
        return "fresh", -1
        return -1
    
    def _updateParentBoudaries(self, parents: pd.DataFrame, parent_level):
        parents['upperLat'] = parents['centerLat'] + self.delta_lat[parent_level-1]
        parents['lowerLat'] = parents['centerLat'] - self.delta_lat[parent_level-1]
        parents['upperLon'] = parents['centerLon'] + self.delta_lon[parent_level-1]
        parents['lowerLon'] = parents['centerLon'] - self.delta_lon[parent_level-1]
        
        return parents


# TODOS:
# start fresh so you can create new parents with no padding
# this is just for one level - complete for all levels
# parents supplied to generate should be an array of levels
# in the main script figure out logic for handling levels and saving to the database
        