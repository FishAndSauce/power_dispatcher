from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Type, List, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
from operator import itemgetter
from matplotlib import pyplot as plt

from grid_resources.technologies import Generator, InstalledGenerator, Storage
from grid_resources.demand import GridDemand
from grid_resources.commodities import Markets
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


def idx(columns, name):
    return list(columns).index(name)


def drop_tuple_if_out_of_bounds(
        arr: List[Tuple[Generator, float]],
        upper_bound: float,
        lower_bound: float
):
    return list([x for x in arr if lower_bound < x[1] < upper_bound])


@dataclass
class Dispatch:
    demand: GridDemand

    def __post_init__(self):
        self.residual_demand = np.array(self.demand.data)
        self.record = pd.DataFrame(
            data=self.demand.data,
            columns=['demand']
        )
        self.rank = []

    def update(
            self,
            dispatch: np.ndarray,
            dispatch_name: str
    ):
        self.residual_demand -= dispatch
        self.record[dispatch_name] = dispatch
        self.rank.append(dispatch_name)
        self.record['residual_demand'] = self.residual_demand

    def plot(self):
        plt_this = []
        rank = self.rank
        rank.append('residual_demand')
        for gen in rank:
            plt_this.append(self.record[gen])

        plt.stackplot(
            self.record.index,
            *plt_this,
            labels=self.rank
        )
        plt.legend()
        plt.show()


@dataclass
class OptimalDeployment:
    ranked_generators: List[InstalledGenerator]

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        pass


@dataclass
class PortfolioOptimiser(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def optimise(
            generators: List[Union[Generator, InstalledGenerator]],
            demand: GridDemand,
    ) -> OptimalDeployment:
        pass


@dataclass
class MeritOrderOptimiser(PortfolioOptimiser):

    @staticmethod
    def optimise(
            generators: List[Generator],
            demand: GridDemand
    ) -> OptimalDeployment:

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
            x['capacity'],
            x['generator']
        ), axis=1).to_list()
        return OptimalDeployment(gen_list)


@dataclass
class ShortRunMarginalCostOptimiser(PortfolioOptimiser):

    @staticmethod
    def optimise(
            generators: List[InstalledGenerator],
            demand: GridDemand,
    ) -> OptimalDeployment:

        ranker = pd.DataFrame(
            index=[g.name for g in generators]
        )
        ranker['generator'] = generators
        ranker['short_run_marginal_cost'] = [
            g.generator.properties.total_var_cost for g in generators
        ]
        ranker.sort_values('short_run_marginal_cost', inplace=True)
        return OptimalDeployment(ranker['generator'].to_list())


@dataclass
class Portfolio(ABC):
    generator_options: List[Union[Generator, InstalledGenerator]]
    storage_options: List[Storage]
    demand: GridDemand
    optimiser: Type[PortfolioOptimiser]
    markets: Markets
    optimal_deployment: OptimalDeployment = None

    def optimise(self):
        self.optimal_deployment = self.optimiser.optimise(
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

    def dispatch(self):
        dispatch = Dispatch(self.demand)
        for generator in self.optimal_deployment.ranked_generators:
            dispatch.update(
                generator.dispatch(dispatch.residual_demand),
                generator.name
            )
        self.dispatched = dispatch

    def plot_dispatch(self):
        if self.dispatched:
            self.dispatched.plot()
        else:
            print('No dispatch has been calculated')


