from dataclasses import dataclass
from typing import List

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
                    asset.cappable_capacity,
                    exceedance
                )
                asset.nameplate_capacity -= capacity_displacement
                exceedance -= capacity_displacement
                if exceedance <= 0:
                    break


@dataclass
class CapacityConstraints(StochasticResource):
    constraints: List[StochasticAnnualCurve]

    def refresh(self):
        for constraint in self.constraints:
            constraint.refresh()
