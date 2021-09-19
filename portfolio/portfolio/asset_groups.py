from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict

import pandas as pd

from portfolio.portfolio.results_logging import DispatchLog
from portfolio.resources.generators import GeneratorTechnology
from portfolio.resources.technologies import Asset


def idx(columns, name):
    return list(columns).index(name)


def drop_tuple_if_out_of_bounds(
        arr: List[Tuple[GeneratorTechnology, float]],
        upper_bound: float,
        lower_bound: float
):
    return list([x for x in arr if lower_bound < x[1] < upper_bound])


@dataclass
class CapacityCapper:
    """ Handler for limiting total capacity of portfolio.
         - Capacity is limited by prioritising some technologies over
        others.
         - Capacity of low priority technology is limited and
        displaced by the aggregate capacity of high priority technology
        in cases where total specified capacity is greater than
        greater than the specified cap.
         - Low priority technologies are ordered by which technology's
         capacity is displaced first
    """
    cappable_assets: List[Asset]

    def cap(self, exceedance: float):
        if exceedance > 0:
            for asset in self.cappable_assets:
                capacity_displacement = min(
                    asset.cappable_capacity * asset.nameplate_capacity,
                    exceedance
                )
                asset.nameplate_capacity -= capacity_displacement
                exceedance -= capacity_displacement
                if exceedance <= 0:
                    break


@dataclass
class AssetGroupOptimiser(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def optimise(
            *args,
            **kwargs
    ) -> List[Asset]:
        pass


# @dataclass
# class MeritOrderOptimiser(AssetGroupOptimiser):
#
#     @staticmethod
#     def optimise(
#             assets: AssetOptions,
#             demand: AnnualCurve
#     ) -> RankedGeneratorDeployment:
#
#         ranker = pd.DataFrame.from_dict(
#             assets.options,
#             'index',
#             columns=['generator']
#         )
#         ranker['rank'] = np.nan
#         ranker['deploy_at'] = np.nan
#         ranker['intercepts'] = np.nan
#         ranker['max_duration_cost'] = np.nan
#         ranker['ranked'] = False
#
#         # Find x intercepts, sorted descending by x value, between all
#         # Find cost of each generators if run for the full period
#         for name, gen in assets.options.items():
#             sorted_intercepts = sorted(
#                 gen.intercept_x_vals(assets),
#                 reverse=True,
#                 key=itemgetter(1)
#             )
#             ranker.at[name, 'intercepts'] = sorted_intercepts
#             ranker.at[name, 'max_duration_cost'] = \
#                 gen.get_period_cost(demand.periods)
#         ranker.sort_values('max_duration_cost', inplace=True)
#
#         # Set gen with lowest max duration cost as 1 in rank column
#         # and set to deploy at any capacity > 0.0
#         ranker.iloc[0, idx(ranker.columns, "rank")] = 1
#         ranker.iloc[0, idx(ranker.columns, "ranked")] = True
#         ranker.iloc[0, idx(ranker.columns, "deploy_at")] = 0.0
#         next_rank = 1
#         # Traverse the break-even envelope, starting at rank 1, to rank each
#         # each generator according to x value of break even points
#         upper_bound = demand.periods
#         lower_bound = 0.0
#         intercepts = drop_tuple_if_out_of_bounds(
#             ranker.iloc[0].loc["intercepts"],
#             upper_bound,
#             lower_bound
#         )
#         next_gen = ranker.iloc[0, idx(ranker.columns, "generator")]
#         while intercepts:
#             # Get next intercept and finalise df data for next_step
#             next_x = intercepts[0][1]
#             demand_at_x = demand.ldc.find_y_at_x(next_x)
#             ranker.at[next_gen.name, "unit_capacity"] = \
#                 demand_at_x - ranker.at[next_gen.name, "deploy_at"]
#
#             # Move on to generator at next intercept
#             next_rank += 1
#             next_gen = intercepts[0][0]
#             upper_bound = next_x
#             # Set next generator rank
#             ranker.at[next_gen.name, "rank"] = next_rank
#             ranker.at[next_gen.name, "ranked"] = True
#             ranker.at[next_gen.name, "deploy_at"] = demand_at_x
#             # get next list of intercepts
#             intercepts = drop_tuple_if_out_of_bounds(
#                 ranker.loc[next_gen.name, "intercepts"],
#                 upper_bound,
#                 lower_bound
#             )
#             if next_rank > len(assets.options) + 1:
#                 raise ValueError(f'There is probably a bug in this loop!'
#                                  f'The number of ranks should not exceed '
#                                  f'the number of generators')
#         # finalise
#         ranker.at[next_gen.name, "unit_capacity"] = \
#             demand.peak - ranker.at[next_gen.name, "deploy_at"]
#
#         ranker = ranker[ranker['ranked']]
#         ranker.sort_values('rank', inplace=True)
#         ranker['capacity'] = ranker['unit_capacity'] * demand.peak
#
#         gen_list = ranker.apply(lambda x: Generator(
#             x['generator'].name,
#             x['capacity'],
#             x['generator']
#         ), axis=1).to_list()
#         return RankedGeneratorDeployment(gen_list)


@dataclass
class ShortRunMarginalCostOptimiser(AssetGroupOptimiser):

    @staticmethod
    def optimise(
            group: RankedAssetGroup,
            optimise_on: str,
    ) -> List[Asset]:
        if not group:
            return None
        ranker = pd.DataFrame.from_dict(
            group.asset_dict,
            'index',
            columns=['technology']
        )
        ranker[optimise_on] = ranker['technology'].apply(
            lambda x: getattr(x.technology, optimise_on)
        )
        ranker.sort_values(optimise_on, inplace=True)
        return ranker['technology'].to_list()


@dataclass
class RankedAssetGroup:
    """ Collection of Assets with ranked dispatch order
    """
    asset_rank: List[Asset]
    optimise_on: str = None

    @property
    def asset_dict(self) -> Dict[str, Asset]:
        return {a.name: a for a in self.asset_rank}

    @property
    def asset_name_list(self) -> List[str]:
        return list([a.name for a in self.asset_rank])

    @property
    def total_capacity(self):
        return sum([
            t.firm_capacity
            for t in self.asset_rank
        ])

    def rank_assets(self, optimiser: AssetGroupOptimiser):
        self.asset_rank = optimiser.optimise(
            self,
            self.optimise_on
        )

    def dispatch(
            self,
            dispatch_logger: DispatchLog,
            log_annual_costs: bool = True,
            log_levelized_cost: bool = True,
    ):
        annual_costs = None
        levelized_cost = None
        for asset in self.asset_rank:
            dispatch = asset.dispatch(
                dispatch_logger.dispatch_log['residual_demand']
            )
            if log_annual_costs:
                annual_costs = asset.annual_dispatch_cost(dispatch),
            if log_levelized_cost:
                levelized_cost = asset.levelized_cost(dispatch)

            dispatch_logger.log(
                dispatch_name=asset.name,
                dispatch=dispatch,
                annual_cost=annual_costs,
                levelized_cost=levelized_cost
            )

    def assets_to_dataframe(
            self,
    ) -> pd.DataFrame:
        assets = pd.DataFrame(self.asset_dict)
        assets['rank'] = assets.index + 1
        return assets

    def update_capacities(self, capacities: dict):
        for gen, new_capacity in capacities.items():
            self.asset_dict[gen].nameplate_capacity = new_capacity


@dataclass
class AssetGroups:
    generators: RankedAssetGroup
    storages: RankedAssetGroup
    passive_generators: RankedAssetGroup
    nominal_capacity_cap: float
    optimiser: ShortRunMarginalCostOptimiser
    capacity_capper: CapacityCapper
    specified_deployment_order: Tuple[str] = ('passive_generators', 'storages', 'generators')
    dispatch_logger: DispatchLog = None

    @property
    def all_assets_name_list(self):
        return \
            self.generators.asset_name_list \
            + self.storages.asset_name_list \
            + self.passive_generators.asset_name_list

    @property
    def all_assets_list(self):
        return \
            self.generators.asset_rank \
            + self.storages.asset_rank \
            + self.passive_generators.asset_rank

    @property
    def all_assets_dict(self) -> Dict[str, Asset]:
        return {
            **self.generators.asset_dict,
            **self.storages.asset_dict,
            **self.passive_generators.asset_dict,
        }

    @property
    def ordered_deployment(self) -> List[RankedAssetGroup]:
        return list([
            getattr(self, tech)
            for tech in self.specified_deployment_order
        ])

    @property
    def total_capacity(self):
        return sum([
            self.generators.total_capacity,
            self.storages.total_capacity,
            self.passive_generators.total_capacity,
        ])

    @property
    def capacity_exceedance(self):
        return max(
            0.0,
            self.total_capacity - self.nominal_capacity_cap
        )

    def cap_capacities(
        self,
    ):
        self.capacity_capper.cap(self.capacity_exceedance)

    def update_capacities(
            self,
            assets: Dict[str, float],
            cap_capacities: bool
    ):
        for name, capacity in assets.items():
            self.all_assets_dict[name].nameplate_capacity = capacity
        if cap_capacities:
            self.capacity_capper.cap(self.capacity_exceedance)

    def asset_capacities(self) -> Dict[str, float]:
        return {
            asset.name: asset.firm_capacity
            for asset in self.all_assets_list
        }

    def assets_to_dataframe(self):
        return pd.concat(
            [assets.assets_to_dataframe()
             for assets in self.ordered_deployment],
            axis=1
        )

    def optimise_groups(self):
        for asset_group in self.ordered_deployment:
            asset_group.rank_assets(self.optimiser)

    def dispatch(
            self,
            demand,
            log_annual_costs: bool = True,
            log_levelized_cost: bool = True,
    ):
        if not self.dispatch_logger:
            self.dispatch_logger = DispatchLog(demand)
        for asset_group in self.ordered_deployment:
            asset_group.dispatch(
                self.dispatch_logger,
                log_annual_costs,
                log_levelized_cost,
            )


