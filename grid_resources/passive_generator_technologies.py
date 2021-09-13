from abc import ABC
from dataclasses import dataclass
from typing import Union, List

import numpy as np

from grid_resources.dynamics import DynamicResource
from grid_resources.technologies import (
    InstalledTechnology,
    GridTechnology,
    TechnoEconomicProperties
)
from grid_resources.curves import AnnualCurve


@dataclass
class PassiveResources(DynamicResource):
    resources: List[AnnualCurve]

    def refresh(self):
        for resource in self.resources:
            resource.refresh()


@dataclass
class PassiveGeneration(ABC):
    generation: np.ndarray


@dataclass
class PassiveTechnoEconomicProperties(TechnoEconomicProperties):
    round_trip_efficiency: float
    levelized_cost: float = None

    @property
    def total_var_cost(self) -> float:
        return self.variable_om


@dataclass
class PassiveTechnology(GridTechnology):
    properties: PassiveTechnoEconomicProperties


@dataclass
class PassiveInstalledGenerator(InstalledTechnology):
    technology: PassiveTechnology
    passive_resource: AnnualCurve
    constraint: Union[float, np.ndarray] = None

    def dispatch(
            self,
            demand: np.ndarray
    ) -> np.ndarray:
        if self.constraint:
            constraint = np.clip(self.passive_resource.data, 0, self.constraint)
        else:
            constraint = self.passive_resource.data

        return np.clip(
            demand,
            0,
            constraint
        )

    def annual_dispatch_cost(self, dispatch: np.ndarray) -> float:
        total_dispatch = dispatch.sum()
        return total_dispatch * self.technology.properties.total_var_cost + \
            self.capacity * self.technology.properties.total_fixed_cost

    def levelized_cost(
            self,
            dispatch: np.ndarray,
            total_dispatch_cost: float = None
    ) -> float:
        if self.technology.properties.levelized_cost:
            return self.technology.properties.levelized_cost
        else:
            if not total_dispatch_cost:
                total_dispatch_cost = self.annual_dispatch_cost(dispatch)
            return total_dispatch_cost / dispatch.sum()

    def hourly_dispatch_cost(
            self,
            dispatch: np.ndarray,
            total_dispatch_cost: float = None,
            levelized_cost: float = None,
    ) -> np.ndarray:
        """ Get hourly dispatch cost based on lcoe
        """
        if not total_dispatch_cost:
            total_dispatch_cost = self.annual_dispatch_cost(dispatch)
        if not levelized_cost:
            levelized_cost = self.levelized_cost(dispatch, total_dispatch_cost)
        return dispatch * levelized_cost
