from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Type, List, Union

from grid.deployment_optimisers import DeploymentOptimiser
from grid_resources.dispatchable_generator_technologies import GeneratorTechnology, InstalledGenerator
from grid_resources.storage_technologies import InstalledStorage
from grid_resources.demand import AnnualDemand
from grid_resources.commodities import Markets
from grid_resources.dispatch import RankedGeneratorDeployment, RankedStorageDeployment, DispatchLogger
from utils.geometry import Lines


optimal_install_df_columns = [
    'rank',
    'capacity',
    'generator'
]


class Validator:
    @staticmethod
    def df_columns(
            df: pd.DataFrame,
            mandatory_columns: list
    ):
        valid = all([col in df.columns for col in mandatory_columns])
        if not valid:
            raise ValueError(
                f'DataFrame not valid: DataFrame columns'
                f' must include all of the following: {", ".join(mandatory_columns)}'
            )


@dataclass
class ScenarioLogger:
    pass


@dataclass
class Portfolio:
    generator_options: List[Union[GeneratorTechnology, InstalledGenerator]]
    storage_options: List[InstalledStorage]
    demand: AnnualDemand
    optimiser: Type[DeploymentOptimiser]
    markets: Markets
    optimal_generator_deployment: RankedGeneratorDeployment = None
    optimal_storage_deployment: RankedStorageDeployment = None
    deploy_storage_first: bool = True
    dispatch_logger: DispatchLogger = None

    def __post_init__(self):
        self.dispatch_logger = DispatchLogger(self.demand)

    @staticmethod
    def build_portfolio(
        generator_options: List[Union[GeneratorTechnology, InstalledGenerator]],
        storage_options: List[InstalledStorage],
        demand: AnnualDemand,
        optimiser: Type[DeploymentOptimiser],
        markets: Markets,
        optimise=True
    ):
        portfolio = Portfolio(
            generator_options,
            storage_options,
            demand,
            optimiser,
            markets,
        )
        if optimise:
            portfolio.optimise()
        return portfolio

    def optimise(self):
        self.optimal_generator_deployment = self.optimiser.optimise(
            self.generator_options,
            self.demand
        )

    def update_markets(self, reoptimise=True):
        self.markets.update_prices()
        if reoptimise:
            self.optimise()

    def plot_ldc(self):
        self.demand.plot_ldc()

    def plot_cost_curves(self):
        my_lines = Lines(
            [g.annual_cost_curve for g in self.generator_options]
        )
        my_lines.plot(0, 8760)

    def dispatch(
            self,
            log_annual_costs: bool = False,
            log_lcoe: bool = False,
            log_hourly_cost: bool = False,
    ):
        deployment_order = [
            self.optimal_storage_deployment,
            self.optimal_generator_deployment
        ]
        if not self.deploy_storage_first:
            deployment_order.reverse()
        for deployment in deployment_order:
            if deployment:
                deployment.dispatch(
                    self.dispatch_logger,
                    log_annual_costs,
                    log_lcoe,
                    log_hourly_cost,
                )

    def plot_dispatch(self):
        self.dispatch_logger.plot()

    def update_demand(self):
        self.demand.update()
        self.dispatch_logger.refresh_log()
