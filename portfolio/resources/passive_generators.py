from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from portfolio.resources.dispatch import DispatchVector
from portfolio.statistics.stochastics import StochasticResource
from portfolio.resources.technologies import (
    Asset,
    GridTechnology,
)
from portfolio.resources.annual_curves import StochasticAnnualCurve, StochasticComplementaryChoiceAnnualCurve


@dataclass
class PassiveResource(StochasticResource):
    resource: StochasticAnnualCurve = None
    data: np.ndarray = None

    def __post_init__(self):
        self.refresh()

    @abstractmethod
    def refresh(self):
        pass


@dataclass
class SimplePassiveResource(PassiveResource):
    resource: StochasticAnnualCurve = None
    data: np.ndarray = None

    def refresh(self):
        self.resource.refresh()
        self.data = np.array(self.resource.data)


@dataclass
class CorrelatedPassiveResource(PassiveResource):
    name: str = None

    def refresh(self):
        self.resource.refresh()
        self.data = np.array(self.resource.data[self.name])


@dataclass
class PassiveResources(StochasticResource):
    resources: List[StochasticAnnualCurve]

    def refresh(self):
        for resource in self.resources:
            resource.refresh()


@dataclass
class PassiveTechnology(GridTechnology):
    levelized_cost: float = None

    @property
    def total_var_cost(self) -> float:
        return self.variable_om


@dataclass
class PassiveGenerator(Asset):
    technology: PassiveTechnology
    passive_resource: PassiveResource

    @property
    def generation_curve(self) -> np.ndarray:
        return self.passive_resource.data * self.nameplate_capacity

    def dispatch(
            self,
            demand: np.ndarray
    ) -> DispatchVector:
        if self.constraint:
            max_dispatch = np.clip(self.generation_curve, 0, self.constraint)
        else:
            max_dispatch = self.generation_curve

        discharge = np.clip(
            demand,
            0,
            max_dispatch
        )
        return DispatchVector(
            name=self.name,
            discharge=discharge,
            excess=np.where(max_dispatch > demand, max_dispatch - demand, 0.0)
        )

    def annual_dispatch_cost(self, dispatch: np.ndarray) -> float:
        total_dispatch = dispatch.sum()
        return total_dispatch * self.technology.total_var_cost + \
               self.firm_capacity * self.technology.total_fixed_cost

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
            dispatch_sum = dispatch.sum()
            if dispatch_sum > 0:
                return total_dispatch_cost / dispatch.sum()
            else:
                return np.nan
