import numpy as np
from scipy import stats, integrate
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from typing import List
import calendar
from datetime import date, datetime

from statistics.stochastics import RandomOptionModel


@dataclass
class Validator:
    @staticmethod
    def standard_year(data):
        if len(data) > 8760:
            raise ValueError(f'Data array must be standard length of 8760'
                             f'to represent 1 year of hourly data (non-leap year)')


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
class AnnualDemand(ABC):
    name: str
    units: str
    demand_data: pd.Series
    #TODO: make generic time handler

    @property
    def periods(self) -> int:
        return len(self.demand_data)

    @property
    def peak_demand(self):
        return max(self.demand_data)

    @property
    def min_demand(self):
        return min(self.demand_data)

    @property
    def ldc(self):
        return LoadDurationCurve.from_data(self.demand_data)

    def plot_ldc(self, show=True):
        self.ldc.data.plot()
        if show:
            plt.show()

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame):
        pass

    @abstractmethod
    def update(self):
        pass


@dataclass
class StochasticAnnualDemand(AnnualDemand):
    year: int = None
    demand_data: pd.Series = None
    sample_data: List[list] = None
    scale: float = 1.0
    strip_leap_days: bool = True
    _direct_instantiation: bool = True

    def __post_init__(self):
        if self._direct_instantiation:
            raise Exception(f'You may only instantiate this objects of this class'
                            f'with class methods - e.g. from_array()')
        self.stochastic_model = RandomOptionModel(self.sample_data)
        self.update()

    def update(
            self,
    ):
        index = pd.date_range(
            start=datetime(self.year, 1, 1, 0),
            end=datetime(self.year, 12, 31, 23),
            freq='H'
        )
        if calendar.isleap(self.year) and self.strip_leap_days:
            index = index[index.date != date(self.year, 2, 29)]
        self.demand_data = pd.Series(
            self.scale * self.stochastic_model.generate_samples(),
            index=index
        )

    @classmethod
    def from_array(
            cls,
            name,
            units,
            year: int,
            data: List[list],
            scale=1.0
    ):
        Validator.standard_year(data)
        return cls(
            name,
            units,
            year=year,
            sample_data=data,
            scale=scale,
            _direct_instantiation=False
        )


