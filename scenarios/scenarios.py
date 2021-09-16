from dataclasses import dataclass

from constraints import CapacityConstraints
from grid.portfolios import ShortRunMarginalCostPortfolio
from grid_resources.curves import StochasticAnnualCurve
from grid_resources.commodities import Markets
from grid_resources.passive_generators import PassiveResources

from grid.asset_group_optimisation import ShortRunMarginalCostOptimiser, AssetGroups


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
    optimiser: ShortRunMarginalCostOptimiser
    constraints: CapacityConstraints

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

    def update_scenario(
            self,
            capacities: dict,
    ):
        # Change scenario data
        if capacities:
            self.portfolio.update_capacities(**capacities)
