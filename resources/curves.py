import numpy as np
from scipy import stats, integrate
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Any
import calendar
from datetime import date, datetime

from statistics.stochastics import (
    RandomArrayChoiceModel,
    StochasticModel,
    RandomWindowChoiceModel,
    StochasticResource
)


@dataclass
class Validator:
    @staticmethod
    def not_none(data, name):
        if not data:
            raise ValueError(f'Invalid attribute value: {name} must not be None')

    @staticmethod
    def standard_year(data):
        if len(data) > 8760:
            raise ValueError(f'Data array must be standard length of 8760'
                             f'to represent 1 year of hourly data (non-leap year)')

    @staticmethod
    def annual_hours(data: List[list]):
        for arr in data:
            if 8760 != len(arr):
                raise ValueError(f'Invalid array: Nested arrays must have length 8760'
                                 f'(i.e. they must represent a year worth of hours '
                                 f'(leap year not accepted)')


@dataclass
class RandomAnnualCurveChoice(RandomArrayChoiceModel):
    def __post_init__(self):
        Validator.annual_hours(self.data)


@dataclass
class StochasticAnnualCurveModel(ABC):
    sample_data: Any
    stochastic_model: StochasticModel = None

    @abstractmethod
    def update(self):
        pass


@dataclass
class StochasticWindowAnnualCurveModel(StochasticAnnualCurveModel):
    stochastic_model: RandomWindowChoiceModel = None
    scale: float = 1.0
    hours: int = 8760

    def update(self):
        return self.scale * self.stochastic_model.generate_samples(self.hours)

    @classmethod
    def from_array(cls, sample_data, scale: float = 1.0):
        stochastic_model = RandomWindowChoiceModel(sample_data)
        return cls(
            sample_data=sample_data,
            stochastic_model=stochastic_model,
            scale=scale
        )


@dataclass
class StochasticChoiceAnnualCurveModel(StochasticAnnualCurveModel):
    name: str = None
    year: int = None
    scale: float = 1.0
    strip_leap_days: bool = True

    _direct_instantiation: bool = True

    def __post_init__(self):
        if self._direct_instantiation:
            raise Exception(f'You may only instantiate this objects of this class'
                            f'with class methods - e.g. from_array()')
        self.stochastic_model = RandomAnnualCurveChoice(self.sample_data)
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
        return pd.Series(
            self.scale * self.stochastic_model.generate_samples(),
            index=index
        )

    @classmethod
    def from_array(
            cls,
            name: str,
            year: int,
            sample_data: List[list],
            scale=1.0,
            strip_leap_days: bool = True
    ):
        Validator.standard_year(sample_data)
        return cls(
            sample_data=sample_data,
            name=name,
            year=year,
            scale=scale,
            strip_leap_days=strip_leap_days,
            _direct_instantiation=False
        )


@dataclass
class DurationCurve:
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
        return DurationCurve(data)


@dataclass
class StochasticAnnualCurve(StochasticResource):
    """ Annual time series whose values are generated stochastically
    by some StochasticAnnualCurveModel
    """
    name: str
    units: str
    data: pd.Series
    stochastic_model: StochasticAnnualCurveModel

    def __post_init__(self):
        Validator.not_none(self.data, 'data')

    # TODO: make generic time handler

    @property
    def periods(self) -> int:
        return len(self.data)

    @property
    def peak(self):
        return max(self.data)

    @property
    def min(self):
        return min(self.data)

    @property
    def ldc(self):
        return DurationCurve.from_data(self.data)

    def plot_ldc(self, show=True):
        self.ldc.data.plot()
        if show:
            plt.show()

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame):
        pass

    @abstractmethod
    def refresh(self):
        pass

    @classmethod
    @abstractmethod
    def from_array(
        cls,
        name,
        units,
        sample_data,
        scale,
    ):
        pass


@dataclass
class StochasticWindowAnnualCurve(StochasticAnnualCurve):
    """ Annual time series whose values are generated stochastically
    by randomly slicing from a multi-year sample data
    """
    stochastic_model: StochasticWindowAnnualCurveModel
    _direct_instantiation: bool = True

    def __post_init__(self):
        if self._direct_instantiation:
            raise Exception(f'You may only instantiate this objects of this class'
                            f'with class methods - e.g. from_array()')

    def refresh(self):
        self.data = self.stochastic_model.update()

    @classmethod
    def from_array(
            cls,
            name: str,
            units: str,
            sample_data: np.ndarray,
            scale=1.0,
    ):
        stochastic_model = StochasticWindowAnnualCurveModel.from_array(
            sample_data,
            scale
        )
        return cls(
            name,
            units,
            stochastic_model.update(),
            stochastic_model,
            _direct_instantiation=False
        )


@dataclass
class StochasticChoiceAnnualCurve(StochasticAnnualCurve):
    """Annual time series whose values are generated stochastically
    by randomly selecting one of a selection of arrays wchi each represent a single year
    """
    stochastic_model: StochasticChoiceAnnualCurveModel = None
    _direct_instantiation: bool = True

    def __post_init__(self):
        if self._direct_instantiation:
            raise Exception(f'You may only instantiate this objects of this class'
                            f'with class methods - e.g. from_array()')

    def refresh(self):
        self.data = self.stochastic_model.update()

    @classmethod
    def from_array(
            cls,
            name: str,
            units: str,
            year: int,
            sample_data: List[list],
            scale=1.0,
            strip_leap_days: bool = True
    ):
        stochastic_model = StochasticChoiceAnnualCurveModel.from_array(
            name,
            year,
            sample_data,
            scale,
            strip_leap_days,
        )
        return cls(
            name,
            units,
            stochastic_model.update(),
            stochastic_model,
            _direct_instantiation=False
        )
