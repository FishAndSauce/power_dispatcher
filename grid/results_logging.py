from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from grid_resources.curves import AnnualCurve


@dataclass
class DispatchLog:
    demand: AnnualCurve

    def __post_init__(self):
        self.dispatch = pd.DataFrame(
            data=self.demand.data,
            columns=['demand']
        )
        self.annual_dispatch_costs = {}
        self.levelized_costs = {}
        self.hourly_costs = pd.DataFrame(
            data=self.demand.data,
            columns=['demand']
        )


@dataclass
class DispatchLogger:
    demand: AnnualCurve
    residual_demand: np.ndarray = None
    rank: list = None

    def __post_init__(self):
        self.residual_demand = np.array(self.demand.data)
        self.log = DispatchLog(self.demand)
        self.rank = []

    def log_dispatch(
            self,
            dispatch: np.ndarray,
            dispatch_name: str,
    ):
        self.residual_demand -= dispatch
        self.log.dispatch[dispatch_name] = dispatch
        self.rank.append(dispatch_name)
        self.log.dispatch['residual_demand'] = self.residual_demand

    def log_annual_dispatch_cost(
            self,
            cost: float,
            dispatch_name: str,
    ):
        self.log.annual_dispatch_costs[dispatch_name] = cost

    def log_levelized_cost(
            self,
            lcoe: float,
            dispatch_name: str,
    ):
        self.log.levelized_costs[dispatch_name] = lcoe

    def log_hourly_dispatch_cost(
            self,
            hourly_cost: np.ndarray,
            dispatch_name: str,
    ):
        self.log.hourly_costs[dispatch_name] = hourly_cost

    def plot(self):
        plt_this = []
        rank = self.rank
        rank.append('residual_demand')
        for gen in rank:
            plt_this.append(self.log.dispatch[gen])

        plt.stackplot(
            self.log.dispatch.index,
            *plt_this,
            labels=self.rank
        )
        plt.legend()
        plt.show()

    def refresh_log(self):
        self.residual_demand = np.array(self.demand.data)
        self.log = DispatchLog(self.demand)
        self.rank = []