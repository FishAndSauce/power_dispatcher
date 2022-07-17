from dataclasses import dataclass
from typing import Tuple

from portfolio.portfolio.constraints import CapacityConstraints
from portfolio.portfolio.results_logging.plotting import StackPlotConfig
from portfolio.portfolio.results_logging.results_logging import MonteCarloLog, ScenarioLogger
from portfolio.resources.annual_curves import StochasticAnnualCurve
from portfolio.resources.commodities import Markets
from portfolio.resources.passive_generators import PassiveResources

from portfolio.portfolio.asset_groups import RankOnOptimiser, AssetGroups


@dataclass
class ScenarioManager:
    """
    Manager for portfolio scenario inputs and data updates:
     - A "scenario" represents a specific configuration of portfolio parameters,
     e.g. specific capacities of generators and storage. It does not include changes
     of stochastic data as each scenario may undergo numerous stochastic iterations
     - methods with the "update" prefix change static parameters and hence
     instantiate a new scenario
     - methods with the "refresh" prefix generate new samples of stochastic data
    """
    year: int
    demand: StochasticAnnualCurve
    markets: Markets
    passive_resource: PassiveResources
    portfolio: AssetGroups
    optimiser: RankOnOptimiser
    constraints: CapacityConstraints
    scenario_summary: dict = None
    scenario_logger: ScenarioLogger = None

    def __post_init__(self):
        self.monte_carlo_logger = MonteCarloLog(self.portfolio.asset_capacities())

    def refresh_constraints(self):
        self.constraints.refresh()

    def refresh_demand(self):
        self.demand.refresh()

    def refresh_passive_generation_resource(self):
        self.passive_resource.refresh()

    def refresh_markets(self, reoptimise=True):
        self.markets.refresh()
        if reoptimise:
            self.portfolio.optimise_groups()

    def refresh_all(self):
        for method in dir(self):
            if method.startswith('refresh_') and method != 'refresh_all':
                refresh = getattr(self, method)
                refresh()

    def clear_dispatch_log(self):
        self.portfolio.dispatch_logger.clear_log()

    def update_capacities(self, nominal_capacities: dict, cap_capacities: bool):
        self.portfolio.update_capacities(nominal_capacities, cap_capacities)
        self.monte_carlo_logger.scenario = self.portfolio.asset_capacities()

    def monte_carlo(
        self,
        iterations: int = 100,
        plot_config: StackPlotConfig = None

    ):
        for simulation in range(iterations + 1):
            self.refresh_all()
            self.portfolio.dispatch(
                self.demand.data,
                plot_config=plot_config
            )
            self.monte_carlo_logger.log_simulation(
                self.portfolio.dispatch_logger.annual_cost_totals(),
            )
            self.clear_dispatch_log()

    def monte_carlo_capacity_scenario(
            self,
            scenario_name,
            nominal_capacities: dict,
            capacity_cap: float,
            iterations: int = 100,
            log_stats: Tuple[str] = ('mean', 'std'),
            plot_config: StackPlotConfig = None
    ):
        self.portfolio.nominal_capacity_cap = capacity_cap
        self.scenario_logger = ScenarioLogger()
        self.scenario_summary = nominal_capacities
        self.update_capacities(nominal_capacities, cap_capacities=True)
        self.monte_carlo(
            iterations,
            plot_config=plot_config
        )
        self.scenario_logger.log_scenario(
            self.monte_carlo_logger.aggregated_statistics(scenario_name, log_stats),
        )