from dataclasses import dataclass

from resources.commodities import Emissions


@dataclass
class EmissionsCharacteristics:
    emissions_rate: float
    rate_units: str
    tariff: Emissions