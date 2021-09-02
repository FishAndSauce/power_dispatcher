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
        index = np.searchsorted(self.data.index, x, 'right')
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
        data = data.sort_values(ascending=False, ignore_index=True)
        return LoadDurationCurve(data)


@dataclass
class GridDemand:
    name: str
    units: str
    data: pd.Series
    periods: int = 8760

    @property
    def peak_demand(self):
        return max(self.data)

    @property
    def min_demand(self):
        return min(self.data)

    @property
    def ldc(self):
        return LoadDurationCurve.from_data(self.data)

    def plot_ldc(self, show=True):
        self.ldc.data.plot()
        if show:
            plt.show()
