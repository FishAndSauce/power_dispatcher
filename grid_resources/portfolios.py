import pandas as pd
from dataclasses import dataclass
from typing import Type, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
from operator import itemgetter

from grid_resources.technologies import Generator, Storage
from grid_resources.demand import GridDemand


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


def drop_tuple_if_over(
        arr: List[Tuple[Generator, float]],
        bound: float
):
    return list([x for x in arr if x[1] < bound])


@dataclass
class Dispatched:
    demand: GridDemand

    def __post_init__(self):
        self.residual_demand = np.array(GridDemand.data)
        self.record = pd.DataFrame(
            data=GridDemand.data
        )

    def update(
            self,
            dispatch: np.ndarray,
            dispatch_name: str
    ):
        self.residual_demand -= dispatch
        self.record[dispatch_name] = dispatch


@dataclass(order=True)
class InstalledGenerator:
    deploy_at: float
    capacity: float
    generator: Generator

    @property
    def name(self):
        return self.generator.name

    @property
    def stop_dispatch_at(self):
        return self.capacity + self.deploy_at

    def dispatch(self, demand: np.ndarray) -> np.ndarray:
        return np.clip(
            demand,
            self.deploy_at,
            self.stop_dispatch_at
        )


@dataclass
class OptimalInstallations:
    installed_generators: List[InstalledGenerator]

    @property
    def deploy_order(self):
        return sorted(self.installed_generators)

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        Validator.df_columns(
            df,
            [
                'deploy_at',
                'capacity',
                'generator'
            ]
        )
        return df.apply(lambda x: InstalledGenerator(
            x['deploy_at'],
            x['capacity'],
            x['generator']
        ), axis=1).to_list


class PortfolioOptimiser(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def optimise(
            generators: List[Generator],
            demand: GridDemand
    ) -> OptimalInstallations:
        pass


@dataclass
class MeritOrderOptimiser(PortfolioOptimiser):

    @staticmethod
    def optimise(
            generators: Tuple[Generator],
            demand: GridDemand
    ) -> OptimalInstallations:

        ranker = pd.DataFrame(
            columns=['rank', 'intercepts', 'max_duration_cost', 'deploy_at'],
            index=[x.name for x in generators]
        )
        ranker['generator'] = generators

        # Find x intercepts, sorted descending by x value, between all
        # generator cost curves (excludes where x <= 0, x >= 1.0)
        # Find cost of each generators if run for the full period
        for gen in generators:
            sorted_intercepts = sorted(
                gen.intercept_x_vals(generators),
                reverse=True,
                key=itemgetter(1)
            )
            ranker.loc[gen.name, 'intercepts'] = sorted_intercepts
            ranker.loc[gen.name, 'max_duration_cost'] = gen.get_period_cost(1.0)
        ranker.sort_values('max_duration_cost', inplace=True)

        # Set gen with lowest max duration cost as 1 in rank column
        # and set to deploy at any capacity > 0.0
        ranker.iloc[0, idx(ranker.columns, "rank")] = 1
        ranker.iloc[0, idx(ranker.columns, "deploy_at")] = 0.0
        next_rank = 1
        # Traverse the break-even envelope, starting at rank 1, to rank each
        # each generator according to x value of break even points
        bound = 1.0
        intercepts = drop_tuple_if_over(
            ranker.iloc[0].loc["intercepts"],
            bound
        )
        while intercepts:
            next_rank += 1
            next_gen, next_x = intercepts[0]
            # Set next generator rank
            ranker.loc[next_gen.name, "rank"] = 1
            ranker.loc[next_gen.name, "deploy_at"] = bound = demand.ldc.find_y_at_x(next_x)
            # get next list of intercepts
            intercepts = drop_tuple_if_over(
                ranker.loc[next_gen, "intercepts"],
                bound
            )
            if next_rank > len(generators) + 1:
                raise ValueError(f'There is probably a bug in this loop!'
                                 f'The number of ranks should not exceed '
                                 f'the number of generators')

        ranker.sort_values('max_duration_cost', inplace=True)
        return OptimalInstallations.from_dataframe(ranker)


class Portfolio(ABC):
    optimiser: Type[PortfolioOptimiser]
    generator_options: List[Generator]
    storage_options: List[Storage]
    demand: GridDemand

    @property
    def optimal_installations(self) -> OptimalInstallations:
        return self.optimiser.optimise(
            self.generator_options,
            self.demand
        )

    def dispatch(self):
        dispatched = Dispatched(self.demand)
        for generator in self.optimal_installations.deploy_order:
            dispatched.update(
                generator.dispatch(self.demand.data.to_numpy()),
                generator.name
            )
        return dispatched
