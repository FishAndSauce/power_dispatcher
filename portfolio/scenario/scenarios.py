from dataclasses import dataclass
from typing import Tuple

from portfolio.portfolio.constraints import CapacityConstraints
from portfolio.portfolio.results_logging import MonteCarloLog, ScenarioLogger
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
    monte_carlo_logger: MonteCarloLog = None
    scenario_logger: ScenarioLogger = None

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

    def monte_carlo(
        self,
        scenario: dict,
        iterations: int = 100,
        inspect_dispatch: bool = False
    ):
        if not self.monte_carlo_logger:
            self.monte_carlo_logger = MonteCarloLog(scenario)
        for simulation in range(iterations + 1):
            self.refresh_all()
            self.portfolio.dispatch(
                self.demand.data,
                inspect_dispatch=inspect_dispatch
            )
            self.monte_carlo_logger.log_simulation(
                self.portfolio.dispatch_logger.annual_cost_totals(),
            )
            self.clear_dispatch_log()

    def monte_carlo_capacity_scenario(
            self,
            scenario_name,
            nominal_capacities: dict,
            iterations: int = 100,
            log_stats: Tuple[str] = ('mean', 'std'),
            inspect_dispatch: bool = False
    ):
        self.scenario_logger = ScenarioLogger()
        self.scenario_summary = nominal_capacities
        self.portfolio.update_capacities(nominal_capacities, cap_capacities=True)
        updated_capacities = self.portfolio.asset_capacities()
        self.monte_carlo(
            updated_capacities,
            iterations,
            inspect_dispatch=inspect_dispatch
        )
        self.scenario_logger.log_scenario(
            self.monte_carlo_logger.aggregated_statistics(scenario_name, log_stats),
        )