from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Type, Dict
import pandas as pd

from statistics.stochastics import CorrelatedDistributionModel


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


class Fuel(Commodity):
    pass


class Emissions(Commodity):
    pass


class PriceModel(ABC):
    commodities: List[Type[Commodity]]

    @abstractmethod
    def update_prices(self):
        pass


class StaticPrice(PriceModel):
    price: str

    def update_prices(self):
        for commodity in self.commodities:
            commodity.price = self.price


@dataclass
class PriceCorrelation(PriceModel):
    correlation_distribution: CorrelatedDistributionModel
    commodities: Dict[str, Commodity]
    name: str

    @staticmethod
    def from_data(
            data: pd.DataFrame,
            commodities: Dict[str],
            distribution: str
    ):
        if not all([x in data.columns for x in commodities.items()]):
            raise ValueError(
                f'All keys in commodities dict must'
                f' be present as headers in data columns'
            )
        data = data[commodities]
        correlation_dist = CorrelatedDistributionModel.from_data(data, distribution)
        instance = PriceCorrelation(correlation_dist, commodities)
        instance.update_price()
        return instance

    def update_prices(self, number_samples=1):
        prices = self.correlation_distribution.generate_samples(
            number_samples
        )
        for commodity_name, commodity in self.commodities:
            commodity.price = prices[commodity_name]


@dataclass
class FuelMarkets:
    market_prices: List[Type[PriceModel]]

    def update_prices(self):
        for market_price in self.market_prices:
            market_price.update_prices()