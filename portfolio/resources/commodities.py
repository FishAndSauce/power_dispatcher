from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

from portfolio.statistics.stochastics import CorrelatedDistributionModel, StochasticResource


class Validator:
    @staticmethod
    def mutually_exclusive(a, b, a_name, b_name):
        if a and b:
            raise ValueError(f'You may specify {a_name} or {b_name}, not both')


@dataclass
class Commodity(ABC):
    name: str
    price: float
    price_units: str


@dataclass
class Fuel(Commodity):
    pass


@dataclass
class Emissions(Commodity):
    pass


@dataclass
class PriceModel(ABC):
    commodities: Dict[str, Commodity]

    @abstractmethod
    def update_prices(self):
        pass


@dataclass
class StaticPrice(PriceModel):

    def update_prices(self):
        pass


@dataclass
class PriceCorrelation(PriceModel):
    correlation_distribution: CorrelatedDistributionModel
    name: str

    def update_prices(self, number_samples=1):
        prices = self.correlation_distribution.generate_samples(
            number_samples
        )
        for name, price in prices.items():
            self.commodities[name].price = round(price, 3)

    @staticmethod
    def from_data(
            data: pd.DataFrame,
            commodities: Dict[str, Commodity],
            distribution: str
    ):
        if not all([k in data.columns for k, v in commodities.items()]):
            raise ValueError(
                f'All keys in commodities dict must'
                f' be present as headers in data columns'
            )
        data = data[commodities]
        correlation_dist = CorrelatedDistributionModel.from_data(data, distribution)
        instance = PriceCorrelation(commodities, correlation_dist, distribution)
        instance.update_prices()
        return instance


@dataclass
class Markets(StochasticResource):
    market_prices: List[PriceModel]

    def refresh(self):
        for market_price in self.market_prices:
            market_price.update_prices()