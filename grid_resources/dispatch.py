from abc import ABC
from dataclasses import dataclass
from typing import List, Union, Tuple

import pandas as pd

from grid_resources.dispatchable_generator_technologies import InstalledGenerator
from grid_resources.storage_technologies import InstalledStorage


@dataclass
class RankedDeployment(ABC):
    ranked_installations: List[Union[InstalledGenerator, InstalledStorage]]

    def dispatch(
            self,
            log_annual_costs: bool = False,
            log_levelized_cost: bool = False,
            log_hourly_cost: bool = False,
            dispatch_logger=None
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
class RankedDeploymentGroup:
    generators: RankedDeployment
    storages: RankedDeployment
    passive_generators: RankedDeployment
    specified_deployment_order: Tuple[str] = ('passive_generators', 'storages', 'generators')
    deployment_order = None

    def __post_init__(self):
        self.deployment_order = list([
            getattr(self, tech)
            for tech in self.specified_deployment_order
        ])
