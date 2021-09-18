from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union
from abc import ABC, abstractmethod
import numpy as np

from portfolio.portfolio.constraints import CapacityConstraint


class Validator:
    @staticmethod
    def is_proportion(value: float, value_name: str):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'{value_name} must be between 0.0 and 1.0')

    @staticmethod
    def cappable_less_than_capacity(cappable, capacity):
        if cappable > capacity:
            raise ValueError(f'Cappable capacity must be less than'
                             f'capacity')


@dataclass
class GridTechnology(ABC):
    """ Power generation/storage technology with techno-economic data
    and associated derived properties
    """
    name: str
    resource_class: str
    capital_cost: float
    life: float
    fixed_om: float
    variable_om: float
    interest_rate: float

    @property
    def crf(self) -> float:
        """ A capital recovery factor (CRF) is the ratio of a constant
            annuity to the present value of receiving that annuity
            for a given length of time
        """
        return self.interest_rate * (1 + self.interest_rate) ** self.life \
               / ((1 + self.interest_rate) ** self.life - 1)

    @property
    def annualised_capital(self) -> float:
        """ Annualised capital is the capital cost per capacity
            multiplied by the capital recovery factor
        """
        return self.capital_cost * self.crf

    @property
    def total_fixed_cost(self) -> float:
        """ Finds sum of all annual fixed costs per capacity supplied
            by this resource

        Returns:
            float: Total fixed cost per capacity
        """
        return self.annualised_capital + self.fixed_om


@dataclass(order=True)
class Asset(ABC):
    """ Installed asset(or aggregation of identical assets), of specific technology type,
    capable of power dispatch (including active and passive dispatch)
    """
    name: str
    nameplate_capacity: float
    firm_capacity_factor: float
    technology: GridTechnology
    constraint: Union[CapacityConstraint, None]
    cappable_capacity: float

    def __post_init__(self):
        Validator.is_proportion(
            self.cappable_capacity,
            'cappable_capacity'
        )

    @property
    def firm_capacity(self):
        return self.nameplate_capacity * self.firm_capacity_factor

    @abstractmethod
    def dispatch(
            self,
            demand: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def annual_dispatch_cost(self, dispatch: np.ndarray) -> float:
        pass

    @abstractmethod
    def levelized_cost(
            self,
            dispatch: np.ndarray,
            total_dispatch_cost: float = None
    ) -> float:
        pass

    def hourly_dispatch_cost(
            self,
            dispatch: np.ndarray,
            total_dispatch_cost: float = None,
            levelized_cost: float = None,
    ) -> np.ndarray:
        pass

    def asset_details(
            self,
            details: List[str] = None
    ) -> dict:
        if not details:
            details = ['name', 'technology', 'capacity']
        return {detail: getattr(self, detail) for detail in details}
