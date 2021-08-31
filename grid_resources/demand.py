import numpy as np
from scipy import stats, integrate
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt


@dataclass
class LoadDurationCurve:
    data: pd.Series

    @property
    def max_demand(self):
        return max(self.data)

    @property
    def min_demand(self):
        return min(self.data)

    @property
    def sample_size(self):
        return len(self.data)

    def find_y_at_x(self, x):
        """ Returns y value for a given x value on LDC
            y is demand axis and x is time axis expressed as proportion of
            max/total
        Args:
            x (float): Value of x (time) for which y (demand) will be found
        Returns:
            float: Value of y (demand) at given x (time) value
        """
        # Can only np.searchsorted on an ascending array, so do length
        # minus backwards search
        index = self.sample_size - np.searchsorted(self.data.index[::-1], x, 'right')
        return self.data[index]

    def find_area(self, lower_bound, upper_bound):
        """Integrates (simpsons rule) for a given section of demand_axis
        Args:
            lower_bound (float): Lower limit for integration, expressed as
                proportion of max demand
            upper_bound (float): upper limit for integration, expressed as
                proportion of max demand
        Returns:
            float: Area value, expressed as proportion of a unit square
        """
        min_index = np.searchsorted(self.data, lower_bound)
        max_index = np.searchsorted(self.data, upper_bound)

        # Take slice of ldc data based on y_min and y_max
        y_axis = self.data[min_index: max_index]
        x_axis = self.data.index[min_index: max_index]
        return integrate.trapz(x_axis, y_axis)

    @staticmethod
    def from_data(data: pd.Series):
        """ Instantiate LoadDurationCurve object from a demand curve.
        """
        demand_data = np.array(data)
        max_demand = demand_data.max()

        # Set number of data points equal to demand sample size
        # Ensures adequate precision compared to runtime
        sample_size = len(demand_data)
        # Get cumulative distribution array (time axis)
        cdf = stats.cumfreq(demand_data, numbins=sample_size, defaultreallimits=(0.0, max_demand))
        time_axis_np = cdf[0]
        # Cdf needs inversion as ldc is a flipped version
        # Normalise against sample size to return proportional representation
        time_axis_np = (sample_size - time_axis_np) / sample_size
        # Create demand axis data increments from bin sizes (cdf[2])
        demand_axis_np = (np.arange(sample_size) // 2) * 2.0
        # Normalise against max demand to return proportional representation
        demand_axis_np = demand_axis_np / len(demand_axis_np)
        return LoadDurationCurve(data=pd.Series(
            data=demand_axis_np,
            index=time_axis_np
        ))


@dataclass
class GridDemand:
    name: str
    units: str
    data: pd.Series

    @property
    def peak_demand(self):
        return max(self.data)

    @property
    def min_demand(self):
        return min(self.data)

    @property
    def ldc(self):
        return LoadDurationCurve.from_data(self.data)
