from dataclasses import dataclass
from typing import List
import numpy as np

from resources.curves import StochasticAnnualCurve
from resources.technologies import Asset
from statistics.stochastics import StochasticResource


@dataclass
class CapacityCapper:
    """ Handler for limiting total capacity of portfolio.
         - Capacity is limited by prioritising some technologies over
        others.
         - Capacity of low priority technology is limited and
        displaced by the aggregate capacity of high priority technology
        in cases where total specified capacity is greater than
        greater than the specified cap.
         - Low priority technologies are ordered by which technology's
         capacity is displaced first
    """
    cappable_assets: List[Asset]

    def cap(self, exceedance: float):
        if exceedance > 0:
            for asset in self.cappable_assets:
                capacity_displacement = min(
                    asset.cappable_capacity * asset.nameplate_capacity,
                    exceedance
                )
                asset.nameplate_capacity -= capacity_displacement
                exceedance -= capacity_displacement
                if exceedance <= 0:
                    break


@dataclass
class CapacityConstraint(StochasticResource):
    constraint_model: StochasticAnnualCurve
    as_factor: bool

    @property
    def constraint(self):
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
            StochasticAnnualCurve.from_array(
                name,
                units,
                sample_data,
                scale,
            ),
            factor
        )

    def refresh(self):
        self.constraint_model.refresh()


@dataclass
class CapacityConstraints(StochasticResource):
    constraints: List[CapacityConstraint]

    def refresh(self):
        for constraint in self.constraints:
            constraint.refresh()
