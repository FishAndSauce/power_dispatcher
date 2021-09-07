from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from grid_resources.technologies import GridTechnology, TechnoEconomicProperties, InstalledTechnology
from utils.time_series_utils import Scheduler, Forecaster, PeakAreas


@dataclass
class StorageTechnoEconomicProperties(TechnoEconomicProperties):
    round_trip_efficiency: float

    @property
    def total_var_cost(self) -> float:
        return self.variable_om


@dataclass
class StorageOptimiser(ABC):
    scheduler: Scheduler
    forecaster: Forecaster
    discharge_threshold: float = 0.0
    charge_threshold = 0.0

    @abstractmethod
    def set_limit(
            self,
            dt: datetime,
            demand: pd.Series,
            energy: float
    ):
        pass

    def dispatch_proposal(self, demand_value: float) -> float:
        proposal = self.discharge_threshold - demand_value
        return proposal


@dataclass
class PeakShaveStorageOptimiser:
    scheduler: Scheduler
    forecaster: Forecaster
    discharge_threshold: float = 0.0
    charge_threshold = 0.0

    def set_limit(
            self,
            dt: datetime,
            demand: pd.Series,
            energy: float
    ):
        if self.scheduler.event_due(dt):
            demand_forecast = self.forecaster.look_ahead(demand, dt)
            sorted_arr = np.sort(demand_forecast)
            peak_areas = PeakAreas.cumulative_peak_areas(sorted_arr)
            index = PeakAreas.peak_area_idx(peak_areas, energy)
            proposed_limit = np.flip(sorted_arr)[index]
            self.discharge_threshold = max(self.discharge_threshold, proposed_limit)

    def dispatch_proposal(self, demand_value: float) -> float:
        proposal = self.discharge_threshold - demand_value
        return proposal


@dataclass
class StorageTechnology(GridTechnology):
    properties: StorageTechnoEconomicProperties


@dataclass
class InstalledStorage(InstalledTechnology):
    technology: StorageTechnology
    hours_storage: float
    charge_capacity: float
    optimiser: StorageOptimiser
    state_of_charge: float = 1.0

    @property
    def energy_capacity(self):
        return self.capacity * self.hours_storage

    @property
    def depth_of_discharge(self) -> float:
        return 1.0 - self.state_of_charge

    @property
    def available_energy(self) -> float:
        return self.state_of_charge * self.energy_capacity

    @property
    def available_storage(self) -> float:
        return self.depth_of_discharge * self.energy_capacity

    def reset_soc(self, new_soc=1.0):
        self.state_of_charge = new_soc

    def update_soc(self, energy):
        """ charge or discharge where positive energy represents charge
        """
        self.state_of_charge += energy / self.energy_capacity

    def update_state(self, energy: float):
        if energy > 0:
            # Apply efficiency on charge only
            energy = self.technology.properties.round_trip_efficiency * energy
        self.state_of_charge += energy / self.energy_capacity

    def energy_request(self, energy) -> float:
        # Negative energy indicates discharge
        # Positive energy indicates charge
        if energy < 0:
            energy_exchange = - min(
                abs(energy),
                self.capacity,
                self.available_energy
            )
        else:
            energy_exchange = min(
                energy,
                self.available_storage,
                self.charge_capacity
            )
        self.update_state(energy_exchange)
        return energy_exchange

    def dispatch(self, demand: pd.Series):

        dispatch = []
        for dt, load_value in demand.iteritems():
            self.optimiser.set_limit(
                dt,
                demand,
                self.available_energy
            )

            dispatch.append(
                self.energy_request(
                    self.optimiser.dispatch_proposal(load_value),
                )
            )
        return np.array(dispatch)

    def annual_dispatch_cost(self, dispatch: np.ndarray) -> float:
        total_dispatch = dispatch.sum()
        return total_dispatch * self.technology.properties.total_var_cost + \
               self.capacity * self.technology.properties.total_fixed_cost

    def levelized_cost(
            self,
            dispatch: np.ndarray,
            total_dispatch_cost: float = None
    ) -> float:
        """ Get levelised cost of energy based on annual dispatch curve
        """
        if not total_dispatch_cost:
            total_dispatch_cost = self.annual_dispatch_cost(dispatch)
        return total_dispatch_cost / dispatch.sum()

    def hourly_dispatch_cost(
            self,
            dispatch: np.ndarray,
            total_dispatch_cost: float = None,
            levelized_cost: float = None,
    ) -> np.ndarray:
        """ Get hourly dispatch cost based on lcoe
        """
        if not total_dispatch_cost:
            total_dispatch_cost = self.annual_dispatch_cost(dispatch)
        if not levelized_cost:
            levelized_cost = self.levelized_cost(dispatch, total_dispatch_cost)
        return dispatch * levelized_cost


