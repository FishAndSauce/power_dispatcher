from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from portfolio.portfolio.results_logging.plotting import StackPlotConfig
from portfolio.resources.dispatch import DispatchVector


@dataclass
class DispatchLog:
    demand: np.ndarray
    dispatch_log: pd.DataFrame = None
    annual_costs: pd.DataFrame = None
    dispatch_order: List[str] = None

    def __post_init__(self):
        self.clear_log(None)

    def clear_log(self, new_demand: np.ndarray = None):
        if new_demand:
            self.demand = new_demand
        self.dispatch_log = pd.DataFrame.from_dict({
            'demand': self.demand,
            'residual_demand': self.demand
        })
        self.annual_costs = pd.DataFrame(
            index=[
                'annual_dispatch_cost',
                'levelized_cost'
            ]
        )
        self.dispatch_order = []

    def log(
        self,
        dispatch: DispatchVector,
        annual_cost: float = None,
        levelized_cost: float = None
    ):
        self.dispatch_log['residual_demand'] -= dispatch.as_net
        self.dispatch_log[dispatch.name] = dispatch.as_net
        self.dispatch_order.append(dispatch.name)
        if annual_cost:
            self.annual_costs.loc[
                'annual_dispatch_cost',
                dispatch.name
            ] = annual_cost
        if levelized_cost:
            self.annual_costs.loc[
                'levelized_cost',
                dispatch.name
            ] = levelized_cost

    def plot(self, plot_config: StackPlotConfig):
        rank = self.dispatch_order
        rank.append('residual_demand')
        plt_this = list([self.dispatch_log[gen] for gen in rank])
        colors = list([plot_config.color_map.get(gen, 'red') for gen in rank])

        plt.stackplot(
            self.dispatch_log.index,
            *plt_this,
            labels=self.dispatch_order,
            colors=colors
        )
        plt.legend()
        plt.show()

    def annual_cost_totals(self):
        df = self.annual_costs.T
        annual_cost_sum = df['annual_dispatch_cost'].sum()
        weighted_cost = df['annual_dispatch_cost'] * df['levelized_cost']
        levelized_cost = weighted_cost.sum() / annual_cost_sum
        return pd.Series(data={
            'annual_dispatch_cost': annual_cost_sum,
            'levelized_cost': levelized_cost
        })


@dataclass
class MonteCarloLog:
    scenario: dict
    log: pd.DataFrame = None

    def __post_init__(self):
        self.log = pd.DataFrame()

    def clear_log(self):
        self.log = pd.DataFrame()

    def log_simulation(
        self,
        iteration_result: pd.Series
    ):
        self.log = self.log.append(iteration_result, ignore_index=True)

    def plot(self):
        pass

    def aggregated_statistics(
        self,
        scenario_name: str,
        stats: Tuple[str] = ('mean', 'std')
    ):
        scenario_name_s = pd.Series({'scenario_name': scenario_name})
        scenario_s = pd.Series(self.scenario)
        rows = []
        for stat in stats:
            stat_method = getattr(pd.DataFrame, stat)
            statistic_s = stat_method(self.log)
            stat_label_s = pd.Series({'statistic': stat})
            rows.append(scenario_s.copy().append([scenario_name_s, stat_label_s, statistic_s]))
        return pd.DataFrame(rows)


@dataclass
class ScenarioLogger:
    log: pd.DataFrame = None

    def __post_init__(self):
        self.clear_log()

    def clear_log(self):
        self.log = pd.DataFrame()

    def log_scenario(self, scenario_results: pd.DataFrame):
        self.log = pd.concat([
            self.log,
            scenario_results
        ], axis=1)

    def plot(self):
        pass