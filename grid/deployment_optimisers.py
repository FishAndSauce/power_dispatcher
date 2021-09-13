from abc import ABC, abstractmethod
from dataclasses import dataclass
from operator import itemgetter
from typing import List, Union, Tuple, Type
import numpy as np

import pandas as pd

from grid_resources.curves import AnnualCurve
from grid_resources.dispatch import RankedDeploymentGroup, RankedDeployment
from grid_resources.dispatchable_generator_technologies import GeneratorTechnology, InstalledGenerator
from grid_resources.technologies import TechnologyOptions, InstalledTechnologyOptions


def idx(columns, name):
    return list(columns).index(name)


def drop_tuple_if_out_of_bounds(
        arr: List[Tuple[GeneratorTechnology, float]],
        upper_bound: float,
        lower_bound: float
):
    return list([x for x in arr if lower_bound < x[1] < upper_bound])


@dataclass
class DeploymentOptimiser(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def optimise(
            generators: TechnologyOptions,
    ) -> RankedGeneratorDeployment:
        pass


@dataclass
class MeritOrderOptimiser(DeploymentOptimiser):

    @staticmethod
    def optimise(
            generators: TechnologyOptions,
            demand: AnnualCurve
    ) -> RankedGeneratorDeployment:
        
        ranker = pd.DataFrame.from_dict(
            generators.options, 
            'index', 
            columns=['generator']
        )
        ranker['rank'] = np.nan
        ranker['deploy_at'] = np.nan
        ranker['intercepts'] = np.nan
        ranker['max_duration_cost'] = np.nan
        ranker['ranked'] = False

        # Find x intercepts, sorted descending by x value, between all
        # Find cost of each generators if run for the full period
        for name, gen in generators.options.items():
            sorted_intercepts = sorted(
                gen.intercept_x_vals(generators),
                reverse=True,
                key=itemgetter(1)
            )
            ranker.at[name, 'intercepts'] = sorted_intercepts
            ranker.at[name, 'max_duration_cost'] = \
                gen.get_period_cost(demand.periods)
        ranker.sort_values('max_duration_cost', inplace=True)

        # Set gen with lowest max duration cost as 1 in rank column
        # and set to deploy at any capacity > 0.0
        ranker.iloc[0, idx(ranker.columns, "rank")] = 1
        ranker.iloc[0, idx(ranker.columns, "ranked")] = True
        ranker.iloc[0, idx(ranker.columns, "deploy_at")] = 0.0
        next_rank = 1
        # Traverse the break-even envelope, starting at rank 1, to rank each
        # each generator according to x value of break even points
        upper_bound = demand.periods
        lower_bound = 0.0
        intercepts = drop_tuple_if_out_of_bounds(
            ranker.iloc[0].loc["intercepts"],
            upper_bound,
            lower_bound
        )
        next_gen = ranker.iloc[0, idx(ranker.columns, "generator")]
        while intercepts:
            # Get next intercept and finalise df data for next_step
            next_x = intercepts[0][1]
            demand_at_x = demand.ldc.find_y_at_x(next_x)
            ranker.at[next_gen.name, "unit_capacity"] =\
                demand_at_x - ranker.at[next_gen.name, "deploy_at"]

            # Move on to generator at next intercept
            next_rank += 1
            next_gen = intercepts[0][0]
            upper_bound = next_x
            # Set next generator rank
            ranker.at[next_gen.name, "rank"] = next_rank
            ranker.at[next_gen.name, "ranked"] = True
            ranker.at[next_gen.name, "deploy_at"] = demand_at_x
            # get next list of intercepts
            intercepts = drop_tuple_if_out_of_bounds(
                ranker.loc[next_gen.name, "intercepts"],
                upper_bound,
                lower_bound
            )
            if next_rank > len(generators.options) + 1:
                raise ValueError(f'There is probably a bug in this loop!'
                                 f'The number of ranks should not exceed '
                                 f'the number of generators')
        # finalise
        ranker.at[next_gen.name, "unit_capacity"] = \
            demand.peak - ranker.at[next_gen.name, "deploy_at"]

        ranker = ranker[ranker['ranked']]
        ranker.sort_values('rank', inplace=True)
        ranker['capacity'] = ranker['unit_capacity'] * demand.peak

        gen_list = ranker.apply(lambda x: InstalledGenerator(
            x['generator'].name,
            x['capacity'],
            x['generator']
        ), axis=1).to_list()
        return RankedGeneratorDeployment(gen_list)


@dataclass
class ShortRunMarginalCostOptimiser(DeploymentOptimiser):

    @staticmethod
    def rank_technologies(
            technology: InstalledTechnologyOptions,
            optimise_against: str,
    ):
        if not technology:
            return None

        ranker = pd.DataFrame.from_dict(
            technology.options,
            'index',
            columns=['technology']
        )
        ranker[optimise_against] = [
            getattr(t.technology.properties, optimise_against)
            for t in technology.options.values()
        ]
        ranker.sort_values('short_run_marginal_cost', inplace=True)
        return RankedDeployment(ranker['generator'].to_list())

    def optimise(
            self,
            generators: InstalledTechnologyOptions = None,
            passive_generators: InstalledTechnologyOptions = None,
            storages: InstalledTechnologyOptions = None
    ) -> RankedDeploymentGroup:
        ranked_generators = self.rank_technologies(generators, 'total_var_cost')
        ranked_storages = self.rank_technologies(storages, 'levelised_cost')
        ranked_passive_generators = self.rank_technologies(passive_generators, 'levelised_cost')

        return RankedDeploymentGroup(
            ranked_generators,
            ranked_storages,
            ranked_passive_generators,
        )