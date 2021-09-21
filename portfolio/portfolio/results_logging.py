from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
        dispatch_name: str,
        dispatch: np.ndarray,
        annual_cost: float = None,
        levelized_cost: float = None
    ):
        self.dispatch_log['residual_demand'] -= dispatch
        self.dispatch_log[dispatch_name] = dispatch
        self.dispatch_order.append(dispatch_name)
        if annual_cost:
            self.annual_costs.loc[
                'annual_dispatch_cost',
                dispatch_name
            ] = annual_cost
        if levelized_cost:
            self.annual_costs.loc[
                'levelized_cost',
                dispatch_name
            ] = levelized_cost

    def plot(self):
        plt_this = []
        rank = self.dispatch_order
        rank.append('residual_demand')
        for gen in rank:
            plt_this.append(self.dispatch_log[gen])

        plt.stackplot(
            self.dispatch_log.index,
            *plt_this,
            labels=self.dispatch_order
        )
        plt.legend()
        plt.show()

    def annual_cost_totals(self):
        return self.annual_costs.sum(axis=1)


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