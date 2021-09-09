from abc import ABC
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from grid_resources.demand import AnnualDemand
from grid_resources.dispatchable_generator_technologies import InstalledGenerator
from grid_resources.storage_technologies import InstalledStorage


@dataclass
class DispatchLog:
    demand: AnnualDemand

    def __post_init__(self):
        self.dispatch = pd.DataFrame(
            data=self.demand.demand_data,
            columns=['demand']
        )
        self.annual_dispatch_costs = {}
        self.levelized_costs = {}
        self.hourly_costs = pd.DataFrame(
            data=self.demand.demand_data,
            columns=['demand']
        )


@dataclass
class DispatchLogger:
    demand: AnnualDemand
    residual_demand: np.ndarray = None
    rank: list = None

    def __post_init__(self):
        self.residual_demand = np.array(self.demand.demand_data)
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
        self.residual_demand = np.array(self.demand.demand_data)
        self.log = DispatchLog(self.demand)
        self.rank = []



@dataclass
class RankedDeployment(ABC):
    ranked_installations: List[Union[InstalledGenerator, InstalledStorage]]

    def dispatch(
            self,
            dispatch_logger,
            log_annual_costs: bool = False,
            log_levelized_cost: bool = False,
            log_hourly_cost: bool = False,
    ):
        for installation in self.ranked_installations:
            dispatch = installation.dispatch(
                dispatch_logger.residual_demand
            )
            dispatch_logger.log_dispatch(
                dispatch,
                installation.name
            )
            if log_annual_costs:
                dispatch_logger.log_annual_dispatch_cost(
                    installation.annual_dispatch_cost(dispatch),
                    installation.name
                )
            if log_levelized_cost:
                dispatch_logger.log_levelized_cost(
                    installation.levelized_cost(
                        dispatch,
                    ),
                    installation.name
                )
            if log_hourly_cost:
                dispatch_logger.log.hourly_costs(
                    installation.hourly_dispatch_cost(
                        dispatch
                    ),
                    installation.name
                )

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        pass

    def installation_details(
            self,
            details: List[str] = None
    ) -> pd.DataFrame:
        installed_dict = [
            g.installation_details(details)
            for g in self.ranked_installations
        ]
        installed = pd.DataFrame(installed_dict)
        installed['rank'] = installed.index + 1
        return installed


@dataclass
class RankedGeneratorDeployment(RankedDeployment):
    ranked_installations: List[InstalledGenerator]


@dataclass
class RankedStorageDeployment(RankedDeployment):
    ranked_installations: List[InstalledStorage]


