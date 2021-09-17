from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union, List, Tuple

import numpy as np

from resources.commodities import Fuel
from resources.technologies import (
    GridTechnology,
    Asset
)
from resources.emissions import EmissionsCharacteristics
from scenarios.constraints import CapacityConstraint
from utils.geometry import Line


@dataclass
class GeneratorTechnology(GridTechnology):
    """ Dispatchable generation technology - e.g. hydro,
    coal, gas, diesel - to which specific annual cost calculations
    are applicable (e.g. cost curves)
    """
    thermal_efficiency: float
    max_capacity_factor: float
    carbon_capture: float
    emissions: EmissionsCharacteristics
    fuel: Fuel

    @property
    def fuel_cost_per_energy(self):
        return self.fuel.price / self.thermal_efficiency

    @property
    def total_var_cost(self) -> float:
        return self.variable_om + \
               self.emissions.tariff.price + \
               self.fuel_cost_per_energy

    @property
    def annual_cost_curve(self) -> Line:
        """Get linear cost curve based on total var and fixed annual costs
        """
        return Line(
            self.total_var_cost,
            self.total_fixed_cost,
            name=self.name
        )

    def get_period_cost(self, period) -> float:
        """ Returns the unit cost per capacity of a resource running
            over a period of time (expressed as years)
        """
        return self.annual_cost_curve.find_y_at_x(period)

    def intercept_x_vals(
            self,
            other_generators: List[GeneratorTechnology]
    ) -> List[Tuple[GeneratorTechnology, float]]:
        """
        Finds the x-coordinates of intercepts between self and another Lines
        Only between 0 and 1 years
        Parallel lines have no intercept
        """
        intercept_list = list()
        for generator in other_generators:
            intercept = self.annual_cost_curve.find_intercept_on_line(
                generator.annual_cost_curve
            )
            if intercept.x:
                intercept_list.append((generator, intercept.x))
        return intercept_list

    @staticmethod
    def from_dict(
            name: str,
            data: Dict[str, Union[str, float]],
            fuels: Dict[str, Fuel],
            emissions_tariff,
            interest_rate
    ):
        fuel = fuels[data['fuel']]
        emissions = EmissionsCharacteristics(
            data['emission_rate'],
            'tonnes / MWh',
            emissions_tariff
        )
        del data['fuel']
        del data['emission_rate']

        return GeneratorTechnology(
            name,
            resource_class='generator',
            fuel=fuel,
            emissions=emissions,
            interest_rate=interest_rate,
            **data
        )


@dataclass(order=True)
class Generator(Asset):
    technology: GeneratorTechnology
    constraint: CapacityConstraint = None

    def dispatch(
            self,
            demand: np.ndarray
    ) -> np.ndarray:
        if self.constraint:
            constraint = self.constraint.constraint
            if self.constraint.as_factor:
                constraint = constraint * self.firm_capacity
            ultimate_constraint = np.clip(constraint, 0, self.firm_capacity)
        else:
            ultimate_constraint = self.firm_capacity

        return np.clip(
            demand,
            0,
            ultimate_constraint
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
        """ Get levelised cost of energy based on annual dispatch curve
        """
        if not total_dispatch_cost:
            total_dispatch_cost = self.annual_dispatch_cost(dispatch)
        return total_dispatch_cost / dispatch.sum()