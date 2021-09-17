from dataclasses import dataclass
from typing import List
import numpy as np
from abc import ABC , abstractmethod

from resources.curves import StochasticAnnualCurve, StochasticWindowAnnualCurve
from statistics.stochastics import StochasticResource


@dataclass
class CapacityConstraint(StochasticResource):
    constraint_model: StochasticAnnualCurve
    as_factor: bool

    @property
    def constraint(self):
        return self.constraint_model.data

    @classmethod
    @abstractmethod
    def from_array(
        cls,
        name,
        units,
        sample_data: np.ndarray,
        factor,
        scale=1.0
    ):
        return cls(
            StochasticAnnualCurve.from_array(
                name,
                units,
                sample_data,
                scale,
            ),
            factor
        )

    @abstractmethod
    def refresh(self):
        self.constraint_model.refresh()


@dataclass
class StochasticWindowCapacityConstraint(CapacityConstraint):
    constraint_model: StochasticWindowAnnualCurve
    as_factor: bool

    def refresh(self):
        return self.constraint_model.data

    @classmethod
    def from_array(
        cls,
        name,
        units,
        sample_data: np.ndarray,
        factor,
        scale=1.0
    ):
        return cls(
            StochasticWindowAnnualCurve.from_array(
                name,
                units,
                sample_data,
                scale,
            ),
            factor
        )



@dataclass
class CapacityConstraints(StochasticResource):
    constraints: List[CapacityConstraint]

    def refresh(self):
        for constraint in self.constraints:
            constraint.refresh()
