from __future__ import annotations

from abc import ABC
import pandas as pd
from dataclasses import dataclass
from typing import List

from grid_resources.dispatchable_generator_technologies import GeneratorTechnology, Generator
from grid_resources.curves import AnnualCurve
from grid_resources.dispatch import AssetGroups
from grid.results_logging import DispatchLogger
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
class Portfolio(ABC):
    demand: AnnualCurve
    asset_groups: AssetGroups

    def plot_ldc(self):
        self.demand.plot_ldc()

    def dispatch(
            self,
            log_annual_costs: bool = False,
            log_lcoe: bool = False,
            log_hourly_cost: bool = False,
            logger: DispatchLogger = None
    ):
        for deployment in self.asset_groups.deployment_order:
            if deployment:
                deployment.dispatch(
                    log_annual_costs,
                    log_lcoe,
                    log_hourly_cost,
                    logger,
                )

    def asset_details(self):
        assets = [
            getattr(self, tech_type)
            for tech_type in self.deploy_order
            if getattr(self, tech_type)
        ]
        details_frames = [i.assets_to_dataframe() for i in installations]
        return pd.concat(
            details_frames,
            axis=1
        )


@dataclass
class ShortRunMarginalCostPortfolio(Portfolio):
    pass


@dataclass
class MeritOrderPortfolio(Portfolio):
    generator_options: List[GeneratorTechnology] = None

    def plot_cost_curves(self):
        my_lines = Lines(
            [g.annual_cost_curve for g in self.generator_options]
        )
        my_lines.plot(0, 8760)
