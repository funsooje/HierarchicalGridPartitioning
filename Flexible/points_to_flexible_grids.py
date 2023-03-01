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

    def generateGrids(self, locations: pd.DataFrame, existingGrids: pd.DataFrame):
        """
        Pairs the locations with the existing grids if they already fall within the grid
        Create new grids if location does not within an existing grid
        Returns an array of all levels of data grid first array containing the original locations
        """

        # Check that locations contains the right columns
        if 'latitude' not in locations.columns:
            raise ValueError("Locations dataframe has no latitude column")
        if 'longitude' not in locations.columns:
            raise ValueError("Locations dataframe has no longitude column")

        parentGrids = pd.DataFrame({'id': pd.Series(dtype='int'),
                                    'centerLon': pd.Series(dtype='float'),
                                    'centerLat': pd.Series(dtype='float'),
                                    'upperLat': pd.Series(dtype='float'),
                                    'lowerLat': pd.Series(dtype='float'),
                                    'upperLon': pd.Series(dtype='float'),
                                    'lowerLon': pd.Series(dtype='float'),
                                    'level': pd.Series(dtype='int')})

        if existingGrids is not None:
            if ('id' not in existingGrids.columns) or ('centerLon' not in existingGrids.columns) or ('centerLat' not in existingGrids.columns) or ('level' not in existingGrids.columns):
                raise ValueError("Existing dataframe should have id, centerLon, centerLat and level columns")
            parentGrids = existingGrids
            parentGrids = self._updateParentBoudaries(parentGrids)


        result = []

        # Level 0
        data_copy = locations.copy()
        data_copy['parentID'] = 0
        data_copy['parent_lon'] = None
        data_copy['parent_lat'] = None
        data_copy['level'] = 0

        # Iterative Method
        data_copy['parentID'] = data_copy.apply(lambda row: self._getParent(row, parentGrids, 1), axis = 1)

        newParents = pd.DataFrame({'id': pd.Series(dtype='int'),
                                    'centerLon': pd.Series(dtype='float'),
                                    'centerLat': pd.Series(dtype='float'),
                                    'upperLat': pd.Series(dtype='float'),
                                    'lowerLat': pd.Series(dtype='float'),
                                    'upperLon': pd.Series(dtype='float'),
                                    'lowerLon': pd.Series(dtype='float'),
                                    'level': pd.Series(dtype='int')})
        for row_id,row in data_copy.iterrows():
            if row['parentID'] > 0:
                continue
            #generate parent for existing
            centerLon, centerLat, newParents = self._createNewParent(row, newParents, 1)
            data_copy.loc[row_id, 'parent_lon'] = centerLon
            data_copy.loc[row_id, 'parent_lat'] = centerLat

        data_copy.parentID = data_copy.parentID.astype('int')
        result.append(data_copy)

        # TODO: repeat for the remaining levels
        # TODO: parent should be given in an array of levels too
        return result


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

    def _getParent(self, row, parents, parent_level):
        if (row['parentID'] > 0):
            return
        lat = row['latitude']
        lon = row['longitude']

        arr = np.where((parents['upperLat'] >= lat) & 
                        (parents['lowerLat'] <= lat) & 
                        (parents['lowerLon'] <= lon) & 
                        (parents['lowerLon'] <= lon) &
                        (parents['level'] == parent_level))[0]
        if len(arr) > 0:
            return parents.loc[arr[0]].id
        else:
            return 0

    def _createNewParent(self, row, parents, parent_level):
        lat = row['latitude']
        lon = row['longitude']
        arr = np.where((parents['upperLat'] >= lat) & 
                    (parents['lowerLat'] <= lat) & 
                    (parents['lowerLon'] <= lon) & 
                    (parents['lowerLon'] <= lon) &
                    (parents['level'] == parent_level))[0]
        if len(arr) > 0:
            parent = parents.loc[arr[0]]
            return parent.centerLon, parent.centerLat, parents
        else:
            # really new parent
            upperLon, lowerLon, upperLat, lowerLat = self.getNewGridParams(lon, lat, parent_level)
            a = {'centerLon': lon, 'centerLat': lat, 'upperLon': upperLon, 'upperLat': upperLat, 'lowerLon': lowerLon, 'lowerLat': lowerLat, 'level': parent_level}
            parents = pd.concat([parents, pd.DataFrame(a, index = [0])], ignore_index=True)
            return lon, lat, parents


    def getNewGridParams(self, lon, lat, level):
        upperLat = lat + self.delta_lat[level-1]
        lowerLat = lat - self.delta_lat[level-1]
        upperLon = lon + self.delta_lon[level-1]
        lowerLon = lon - self.delta_lon[level-1]
        return upperLon, lowerLon, upperLat, lowerLat

    
    def _updateParentBoudaries(self, parents: pd.DataFrame):
        parents['upperLat'] = parents.centerLat + self.delta_lat[0]
        parents['lowerLat'] = parents.centerLat - self.delta_lat[0]
        parents['upperLon'] = parents.centerLon + self.delta_lon[0]
        parents['lowerLon'] = parents.centerLon - self.delta_lon[0]
        for i in range(1, self.levels):
            parents['upperLat'] = np.where(parents.level == i+1, parents.centerLat + self.delta_lat[i], parents['upperLat'])
            parents['lowerLat'] = np.where(parents.level == i+1, parents.centerLat + self.delta_lat[i], parents['lowerLat'])
            parents['upperLon'] = np.where(parents.level == i+1, parents.centerLon + self.delta_lat[i], parents['upperLon'])
            parents['lowerLon'] = np.where(parents.level == i+1, parents.centerLon + self.delta_lat[i], parents['lowerLon'])

        return parents



# FORMER method
# run through batch and find matches
# run through batch and create new parents
# push found batches and parents in one go
# rerun to link unfound batches with pushed parents
        