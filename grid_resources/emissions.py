from dataclasses import dataclass

from grid_resources.commodities import Emissions


@dataclass
class EmissionsCharacteristics:
    emissions_rate: float
    rate_units: str
    tariff: Emissions