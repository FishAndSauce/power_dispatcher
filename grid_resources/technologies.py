from __future__ import annotations

from dataclasses import dataclass
from typing import Type, List, Tuple, Dict, Union
from abc import ABC

from utils.geometry import Line
from grid_resources.commodities import Fuel, Emissions


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
        """ Finds sum of all fixed costs per capacity supplied
            by this resource

        Returns:
            float: Total fixed cost per capacity
        """
        return self.annualised_capital + self.fixed_om


@dataclass
class GeneratorTechnoEconomicProperties(TechnoEconomicProperties):
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
        return self.variable_om +\
               self.emissions.tariff.price + \
               self.fuel_cost_per_energy

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

        return GeneratorTechnoEconomicProperties(
            name,
            resource_class='generator',
            fuel=fuel,
            emissions=emissions,
            interest_rate=interest_rate,
            **data
        )


@dataclass
class StorageTechnoEconomicProperties(TechnoEconomicProperties):
    lcos: float

    @property
    def total_var_cost(self) -> float:
        return self.variable_om


@dataclass
class GridResource(ABC):
    name: str
    properties: Type[TechnoEconomicProperties]


@dataclass
class Generator(GridResource):
    properties: GeneratorTechnoEconomicProperties

    @property
    def annual_cost_curve(self) -> Line:
        """Get linear cost curve based on total var and fixed annual costs
        """
        return Line(
            self.properties.total_var_cost,
            self.properties.total_fixed_cost,
            name=self.name
        )

    def get_period_cost(self, period) -> float:
        """ Returns the unit cost per capacity of a resource running
            over a period of time (expressed as years)
        """
        return self.annual_cost_curve.find_y_at_x(period)

    def intercept_x_vals(
        self,
        other_generators: Tuple[Generator]
    ) -> List[Tuple[Generator, float]]:
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


@dataclass
class Storage(GridResource):
    properties: StorageTechnoEconomicProperties
    energy_capacity: float
    charge_capacity: float
    discharge_capacity: float
    soc: float = 1.0

    @property
    def dod(self) -> float:
        return 1 - self.soc

    @property
    def available_energy(self) -> float:
        return self.soc * self.energy_capacity

    @property
    def available_storage(self) -> float:
        return self.dod * self.energy_capacity

    def reset_soc(self, new_soc=1):
        self.soc = new_soc

    def update_soc(self, energy):
        """ charge or discharge where positive energy represents charge
        """
        self.soc += energy / self.energy_capacity

    def discharge_request(self, energy_requested):
        """ Responds to request for discharge according to present status of device

        Args:
            energy_requested (float): Amount of energy requested for discharge
        Returns:
            float: Total possible amount of energy that can be discharged, up to the amount requested
        """

        # If there is available energy, discharge
        discharge = min(energy_requested, self.discharge_capacity, self.available_energy)
        self.update_soc(-discharge)
        return discharge

    def charge_request(self, charge_available):
        """ Responds to request to charge according to present status of device
        Args:
            charge_available (float): Amount of energy being offered for charging
        Returns:
            TYPE: Total possible amount of charge that can occur, up to the amount being offered
        """
        charge = min(charge_available, self.charge_capacity, self.available_storage)
        self.update_soc(charge)
        return charge
