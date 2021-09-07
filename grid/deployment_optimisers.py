from abc import ABC, abstractmethod
from dataclasses import dataclass
from operator import itemgetter
from typing import List, Union, Tuple, Type
import numpy as np

import pandas as pd

from grid_resources.demand import AnnualDemand
from grid_resources.dispatch import RankedGeneratorDeployment
from grid_resources.dispatchable_generator_technologies import GeneratorTechnology, InstalledGenerator


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
            generators: List[Union[GeneratorTechnology, InstalledGenerator]],
            demand: AnnualDemand,
    ) -> RankedGeneratorDeployment:
        pass


@dataclass
class MeritOrderOptimiser(DeploymentOptimiser):

    @staticmethod
    def optimise(
            generators: List[GeneratorTechnology],
            demand: AnnualDemand
    ) -> RankedGeneratorDeployment:

        ranker = pd.DataFrame(
            columns=[
                'rank',
                'deploy_at',
                'ranked',
                'intercepts',
                'max_duration_cost',
            ],
            index=[g.name for g in generators]
        )
        ranker['ranked'] = False
        ranker['generator'] = generators

        # Find x intercepts, sorted descending by x value, between all
        # Find cost of each generators if run for the full period
        for gen in generators:
            sorted_intercepts = sorted(
                gen.intercept_x_vals(generators),
                reverse=True,
                key=itemgetter(1)
            )
            ranker.at[gen.name, 'intercepts'] = sorted_intercepts
            ranker.at[gen.name, 'max_duration_cost'] = \
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
            if next_rank > len(generators) + 1:
                raise ValueError(f'There is probably a bug in this loop!'
                                 f'The number of ranks should not exceed '
                                 f'the number of generators')
        # finalise
        ranker.at[next_gen.name, "unit_capacity"] =\
            demand.peak_demand - ranker.at[next_gen.name, "deploy_at"]

        ranker = ranker[ranker['ranked']]
        ranker.sort_values('rank', inplace=True)
        ranker['capacity'] = ranker['unit_capacity'] * demand.peak_demand

        gen_list = ranker.apply(lambda x: InstalledGenerator(
            x['generator'].name,
            x['capacity'],
            x['generator']
        ), axis=1).to_list()
        return RankedGeneratorDeployment(gen_list)


@dataclass
class ShortRunMarginalCostOptimiser(DeploymentOptimiser):

    @staticmethod
    def optimise(
            generators: List[InstalledGenerator],
            demand: AnnualDemand,
    ) -> RankedGeneratorDeployment:

        ranker = pd.DataFrame(
            index=[g.name for g in generators]
        )
        ranker['generator'] = generators
        ranker['short_run_marginal_cost'] = [
            g.technology.properties.total_var_cost for g in generators
        ]
        ranker.sort_values('short_run_marginal_cost', inplace=True)
        return RankedGeneratorDeployment(ranker['generator'].to_list())
