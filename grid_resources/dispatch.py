from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple, Dict, Type
import pandas as pd

from grid.deployment_optimisers import AssetGroupOptimiser, MeritOrderOptimiser
from grid_resources.technologies import Asset


@dataclass
class RankedAssetGroup(ABC):
    """ Collection of Assets with ranked dispatch order
    """
    asset_rank: List[Asset]
    optimiser: AssetGroupOptimiser

    @property
    def asset_dict(self) -> Dict[str, Asset]:
        return {a.name: a for a in self.asset_rank}

    def rank_assets(self):
        self.asset_rank = self.optimiser.optimise(self.asset_rank)

    def dispatch(
            self,
            log_annual_costs: bool = False,
            log_levelized_cost: bool = False,
            log_hourly_cost: bool = False,
            dispatch_logger=None
    ):
        for installation in self.asset_rank:
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

    def assets_to_dataframe(
            self,
    ) -> pd.DataFrame:
        assets = pd.DataFrame(self.asset_dict)
        assets['rank'] = assets.index + 1
        return assets

    def update_capacities(self, capacities: dict):
        for gen, new_capacity in capacities.items():
            self.asset_dict[gen] = new_capacity

    def total_capacity(self):
        return sum([
            t.capacity
            for t in self.asset_rank
        ])


@dataclass
class AssetGroups:
    generators: RankedAssetGroup
    storages: RankedAssetGroup
    passive_generators: RankedAssetGroup
    optimiser: AssetGroupOptimiser
    specified_deployment_order: Tuple[str] = ('passive_generators', 'storages', 'generators')
    deployment_order: List[RankedAssetGroup] = None

    def __post_init__(self):
        self.deployment_order = list([
            getattr(self, tech)
            for tech in self.specified_deployment_order
        ])

    def assets_to_dataframe(self):
        return pd.concat(
            [assets.assets_to_dataframe()
             for assets in self.deployment_order],
            axis=1
        )
    def optimise_groups(self):
        for asset_group in self.deployment_order:
            asset_group.rank_assets()
