"""
Hierarchical Grid Partitioning

This is a script that map points on a map to grids with specific sizes.

Assumptions:
- Measurements are in kilometres (km) - this will be improved later to include calculations with miles
- Grid are squares i.e. a grid size of 0.025 denotes a 25m by 25m base grid size
"""

# Imports
import math
import numpy as np
import pandas as pd

class hierGP:

    def __init__(self, base_size: float, silent: bool = True):
        
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

        self.base_size = base_size / 1000 # specified in meters
        self.levels = 15 # max resolution that can specified 

        self.equatorial_radius = 6378.1370
        self.polar_radius = 6356.7523
        self.lon_per_deg = (self.equatorial_radius * math.pi) / 180
        self.lat_per_deg = (self.polar_radius * math.pi) / 180

        self.epicenterLat = 0
        self.epicenterLon = 0
        self.measurement_sensitivity = 20

        self.delta_lat = []
        self.delta_lon = []
        self._getDeltas(silent)

    
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

    
    def printConfig(self):
        """
        Function for view instance parameters and information
        """
        print("-----------------------------------------")
        print("Hierarchical Grid Partitioning Configuration")
        print("-----------------------------------------")
        print("Base Grid Size: {0}m by {0}m".format(self.base_size * 1000))
        print("Equitorial Radius: {}km".format(self.equatorial_radius))
        print("Polar Radius: {}km".format(self.polar_radius))
        print("-----------------------------------------")

    def generateGrids(self, data: pd.DataFrame, resolution: int, lat: str = "latitude", lon: str = 'longitude'):
        """
        Generate the grids and cell indices.
        Returns an array of all levels of data grid
        """
        resolution = resolution - 1
        data_copy = data.copy()
        lat_array = data_copy[lat].to_numpy()
        lon_array = data_copy[lon].to_numpy()

        midpoints = self._getMidpoints(lat_array, lon_array, resolution)

        data_copy = data_copy.join(midpoints)

        return data_copy

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
        df.columns = ["l{}_lon".format(level+1), "l{}_lat".format(level+1), "l{}_x".format(level+1), "l{}_y".format(level+1)]
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