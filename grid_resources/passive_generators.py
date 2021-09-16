from abc import ABC
from dataclasses import dataclass
from typing import Union, List

import numpy as np

from statistics.stochastics import StochasticResource
from grid_resources.technologies import (
    Asset,
    GridTechnology,
)
from grid_resources.curves import StochasticAnnualCurve


@dataclass
class PassiveResources(StochasticResource):
    resources: List[StochasticAnnualCurve]

    def refresh(self):
        for resource in self.resources:
            resource.refresh()


@dataclass
class PassiveTechnology(GridTechnology):
    round_trip_efficiency: float
    levelized_cost: float = None

    @property
    def total_var_cost(self) -> float:
        return self.variable_om


@dataclass
class PassiveGenerator(Asset):
    technology: PassiveTechnology
    passive_resource: StochasticAnnualCurve
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
        return total_dispatch * self.technology.total_var_cost + \
            self.capacity * self.technology.total_fixed_cost

    def levelized_cost(
            self,
            dispatch: np.ndarray,
            total_dispatch_cost: float = None
    ) -> float:
        if self.technology.levelized_cost:
            return self.technology.levelized_cost
        else:
            if not total_dispatch_cost:
                total_dispatch_cost = self.annual_dispatch_cost(dispatch)
            return total_dispatch_cost / dispatch.sum()
