"""
Points to Grid

This is a script that map points on a map to grids with specific sizes.

Assumptions:
- Measurements are in kilometres (km) - this will be improved later to include calculations with miles
- Grid are squares i.e. a grid size of 0.025 denotes a 25m by 25m base grid size
"""

# Imports
import math
import numpy as np
import pandas as pd

# Global Constants
#

class P2G:

    def __init__(self, base_size: float, levels: int = 1, 
            epicenterLat: float = 0, epicenterLon: float = 0, 
            sensitivity: int = 4, equatorialRadius: float = 6378.1370, 
            polarRadius: float = 6356.7523, silent: bool = True):
        
        """
        Main Constructor

        :param base_size: for base grid size e.g. 0.05 for 50m grids
        :param levels: for number of levels to be built up in the grid system e.g. 10
        :param epicenterLat: for model's center latitude - defaults to 0
        :param epicenterLon: for model's cneter longitude - defaults to 0
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

        self.epicenterLat = epicenterLat
        self.epicenterLon = epicenterLon
        self.measurement_sensitivity = sensitivity

        self.delta_lat = []
        self.delta_lon = []
        self._getDeltas(silent)

    def printConfig(self):
        """
        Function for view instance parameters and information
        """
        print("-----------------------------------------")
        print("Point To Grid Config")
        print("-----------------------------------------")
        print("Base Grid Size: {0}km by {0}km".format(self.base_size))
        print("Grid Levels: {}".format(self.levels))
        print("Equitorial Radius: {}km".format(self.equatorial_radius))
        print("Polar Radius: {}km".format(self.polar_radius))
        print("Model Center: ({}, {})".format(self.epicenterLat, self.epicenterLon))
        print("Base Latitude Delta: {}".format(self.delta_lat[0]))
        print("Base Longitude Delta: {}".format(self.delta_lon[0]))
        print("Actual Grid Size: {}km by {}km".format(np.round(self.lon_per_deg*self.delta_lon[0], 3), np.round(self.lat_per_deg*self.delta_lat[0], 3)))
        print("-----------------------------------------")

    def generateGrids(self, data: pd.DataFrame):
        """
        Generate the grids and cell indices.
        Returns an array of all levels of data grid
        """
        result = []
        data_copy = data.copy()
        lat_array = data_copy['latitude'].to_numpy()
        lon_array = data_copy['longitude'].to_numpy()

        midpoints = self._getMidpoints(lat_array, lon_array)

        data_copy = data_copy.join(midpoints)

        result.append(data_copy)

        prev = data_copy.copy()
        prev = prev[["parent_lon", "parent_lat", "parent_x", "parent_y"]]
        prev = prev.drop_duplicates().reset_index(drop=True)
        prev.columns = ['longitude', 'latitude', 'x', 'y']
        for level in range(1, self.levels):
            lat_array = prev['latitude'].to_numpy()
            lon_array = prev['longitude'].to_numpy()
            midpoints = self._getMidpoints(lat_array, lon_array, level)
            prev = prev.join(midpoints)
            # prev['level'] = level
            result.append(prev)
            if level <= self.levels - 1:
                prev = prev.copy()
                prev = prev[["parent_lon", "parent_lat", "parent_x", "parent_y"]]
                prev = prev.drop_duplicates().reset_index(drop=True)
                prev.columns = ['longitude', 'latitude', 'x', 'y']

        return result

    def generateCenters(self, data: pd.DataFrame, level: int, xcol: str = 'x', ycol: str = 'y'):
        """
        Generate center lon and lat for a given level 
        """
        data_copy = data.copy()
        x_indices = data_copy[xcol].to_numpy().reshape(-1, 1)
        y_indices = data_copy[ycol].to_numpy().reshape(-1, 1)
        cell_indices = np.concatenate((x_indices, y_indices), axis=1)
        df = self._getMidpointsFromIndices(cell_indices=cell_indices, level=level-1)
        data_copy['center_lon'] = df['parent_lon']
        data_copy['center_lat'] = df['parent_lat']
        return data_copy

    def _getDeltas(self, silent: bool):
        """
        Computes lat and lon deltas
        """
        if not silent: 
            print("Base Delta")
            print('Delta Long without dropping decimal places:',self.base_size / self.lon_per_deg)
            print('Delta Lat without dropping decimal places:',self.base_size / self.lat_per_deg)
        self.delta_lon.append(np.round((self.base_size / self.lon_per_deg),self.measurement_sensitivity))
        self.delta_lat.append(np.round((self.base_size / self.lat_per_deg),self.measurement_sensitivity))
        if not silent:
            print('Delta Long After dropping decimal places:',self.delta_lon[0])
            print('Delta Lat After dropping decimal places:',self.delta_lat[0])
            print('Reducing decimal point leads to actual grid size (lon):',self.lon_per_deg*self.delta_lon[0])
            print('Reducing decimal point leads to actual grid size (lat):',self.lat_per_deg*self.delta_lat[0])

        for level in range(1, self.levels):
            level_size = self.base_size * math.pow(2, level)
            self.delta_lon.append(np.round((level_size / self.lon_per_deg),self.measurement_sensitivity))
            self.delta_lat.append(np.round((level_size / self.lat_per_deg),self.measurement_sensitivity))

    def _getMidpoints(self, lat_values: np.array, lon_values: np.array, level: int = 0):
        """
        Computes midpoint of lat and lon arrays
        """
        cell_indices = self._getCellIndices(lat_values, lon_values, level)
        return self._getMidpointsFromIndices(cell_indices=cell_indices, level=level)

    def _getMidpointsFromIndices(self, cell_indices: np.array, level: int):
        midpoints = np.copy(cell_indices.astype(float))

        # lon mid points
        midpoints[:, 0] = (self.delta_lon[level] * (cell_indices[:, 0] + 0.5)) + self.epicenterLon

        # lat mid points
        midpoints[:, 1] = (self.delta_lat[level] * (cell_indices[:, 1] + 0.5)) + self.epicenterLat

        df = pd.DataFrame(
                np.concatenate((midpoints, cell_indices), axis=1), columns=["parent_lon", "parent_lat", "parent_x", "parent_y"])
        df.parent_x = df.parent_x.astype('int32')
        df.parent_y = df.parent_y.astype('int32')
        df['level'] = level
        return df

    def _getCellIndices(self, lat_values: np.array, lon_values: np.array, level: int = 0):
        """
        Computes indices of grid centers
        """
        norm_lon_values = lon_values - self.epicenterLon
        norm_lat_values = lat_values - self.epicenterLat
        x_indices = np.floor(norm_lon_values / self.delta_lon[level]).reshape(-1, 1)
        y_indices = np.floor(norm_lat_values / self.delta_lat[level]).reshape(-1, 1)
        return np.concatenate((x_indices, y_indices), axis=1)
        