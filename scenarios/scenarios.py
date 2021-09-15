from dataclasses import dataclass
from typing import Dict

from grid.portfolios import ShortRunMarginalCostPortfolio
from grid_resources.curves import AnnualCurve
from grid_resources.commodities import Markets
from grid_resources.passive_generator_technologies import PassiveResources

from grid_resources.technologies import (
    RankedAssets,
    Asset,
)
from grid.dispatch_optimisation import AssetOptions, ShortRunMarginalCostOptimiser


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
    capacity_cap: float
    total_capacity: float
    high_priorities: AssetOptions
    low_priorities: RankedAssets
    residual_exceedance: float = 0.0
    
    @property
    def exceedance(self):
        return max([
            0.0,
            self.total_capacity - self.capacity_cap
        ])

    def cap_total_capacity(self):
        residual_exceedance = self.exceedance
        if self.exceedance > 0:
            for tech in self.low_priorities.asset_rank:
                capacity_displacement = min(
                    tech.capacity,
                    residual_exceedance
                )
                tech.capacity -= capacity_displacement
                residual_exceedance -= capacity_displacement
                if residual_exceedance <= 0:
                    break
        self.residual_exceedance = residual_exceedance


@dataclass
class CapacityConstraints:
    constraints: Dict[str, AnnualCurve]


@dataclass
class SRMCScenarioManager:
    """
    Manager for portfolio scenario inputs and updates
    and runner for Monte Carlo simulations:
     - A "scenario" represents a specific configuration of portfolio parameters,
     e.g. specific capacities of generators and storage. It does not include changes
     of stochastic data as each scenario may undergo numerous stochastic iterations
     - methods with the "update" prefix change static parameters and hence
     instantiate a new scenario
     - methods with the "refresh" prefix trigger new samples of stochastic data
     and hence facilitate new Monte Carlo iterations

    """
    year: int
    demand: AnnualCurve
    markets: Markets
    passive_resource: PassiveResources
    generators: AssetOptions
    storages: AssetOptions
    passive_generators: AssetOptions
    capacity_capper: CapacityCapper
    constraints: CapacityConstraints
    optimiser: ShortRunMarginalCostOptimiser
    portfolio: ShortRunMarginalCostPortfolio = None

    def __post_init__(self):
        self.build_portfolio()

    @property
    def all_technologies_list(self):
        return \
            self.generators.asset_list \
            + self.storages.asset_list \
            + self.passive_generators.asset_list

    @property
    def all_technologies_dict(self) -> Dict[str, Asset]:
        return {
            **self.generators.options,
            **self.storages.options,
            **self.passive_generators.options,
        }

    def build_portfolio(self):
        ranked_deployments = self.optimise_deployments()
        return ShortRunMarginalCostPortfolio(
            self.demand,
            ranked_deployments
        )

    def update_capacities(
            self,
            generators: dict,
            storages: dict,
            passive_generators: dict,
    ):
        if generators:
            self.generators.update_capacities(generators)
        if storages:
            self.storages.update_capacities(storages)
        if passive_generators:
            self.passive_generators.update_capacities(passive_generators)
        self.capacity_capper.cap_total_capacity()

    def refresh_constraints(self):
        for name, constraint in self.constraints.constraints.items():
            self.all_technologies_dict[name].constraint = constraint

    def refresh_demand(self):
        self.demand.refresh()

    def refresh_passive_generation_resource(self):
        self.passive_resource.refresh()

    def refresh_markets(self, reoptimise=True):
        self.markets.refresh()
        if reoptimise:
            self.portfolio.optimise()

    def refresh_all(self):
        for method in dir(self):
            if method.startswith('refresh_') and method != 'refresh_all':
                refresh = getattr(self, method)
                refresh()

    def run_monte_carlo(self, iterations=100):
        # For a given scenario, run monte carlo simulations and
        # log results
        for i in range(1, iterations + 1):
            self.refresh_all()
            self.portfolio.dispatch()

    def update_scenario(
            self,
            capacities: dict,
    ):
        # Change scenario data
        if capacities:
            self.update_capacities(**capacities)