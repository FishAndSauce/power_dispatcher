from __future__ import annotations

from dataclasses import dataclass
from typing import Type
from abc import ABC, abstractmethod
import numpy as np

from grid_resources.commodities import Emissions


@dataclass
class EmissionsCharacteristics:
    emissions_rate: float
    rate_units: str
    tariff: Emissions


@dataclass
class TechnoEconomicProperties(ABC):
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


@dataclass
class GridTechnology(ABC):
    name: str
    properties: Type[TechnoEconomicProperties]


@dataclass(order=True)
class InstalledTechnology(ABC):
    name: str
    capacity: float
    technology: GridTechnology

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

