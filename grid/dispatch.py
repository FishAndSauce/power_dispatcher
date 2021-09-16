from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from grid.asset_group_optimisation import AssetGroups
from grid_resources.curves import StochasticAnnualCurve


@dataclass
class DispatchLog:
    demand: np.ndarray
    log: pd.DataFrame = None
    annual_costs: pd.DataFrame = None
    dispatch_order: List[str] = None

    def __post_init__(self):
        self.clear_log(None)

    def clear_log(self, new_demand: np.ndarray = None):
        if new_demand:
            self.demand = new_demand
        self.log = pd.DataFrame.from_dict({
            'demand': self.demand,
            'residual_demand': self.demand
        })
        self.annual_costs = pd.DataFrame(
            index=[
                'annual_dispatch_cost',
                'levelized_cost'
            ]
        )
        self.dispatch_order = []

    def log_dispatch(
        self,
        dispatch_name: str,
        dispatch: np.ndarray,
        annual_cost: float = None,
        levelized_cost: float = None
    ):
        self.log['residual_demand'] -= dispatch
        self.log[dispatch_name] = dispatch
        self.dispatch_order.append(dispatch_name)
        if annual_cost:
            self.annual_costs.loc[
                'annual_dispatch_cost',
                dispatch_name
            ] = annual_cost
        if levelized_cost:
            self.annual_costs.loc[
                'levelized_cost',
                dispatch_name
            ] = levelized_cost

    def plot(self):
        plt_this = []
        rank = self.dispatch_order
        rank.append('residual_demand')
        for gen in rank:
            plt_this.append(self.log.dispatch[gen])

        plt.stackplot(
            self.log.dispatch.index,
            *plt_this,
            labels=self.dispatch_order
        )
        plt.legend()
        plt.show()
