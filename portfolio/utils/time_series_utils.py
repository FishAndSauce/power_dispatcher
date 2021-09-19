from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from dataclasses import dataclass
from time import time
from typing import Tuple


class Timer:
    def __init__(self):
        self.log = {}

    def event_start(self, event_name, metadata=None):
        self.log[event_name] = {
            'start_time': time(),
            'metadata': metadata
        }

    def event_end(self, event_name):
        try:
            time_taken = time() - self.log[event_name]['start_time']
            self.log[event_name]['time_taken'] = time_taken
            print(f'Timer: {event_name} completed in {time_taken} seconds \n'
                  f'metadata: {self.log[event_name]["metadata"]}')
        except KeyError:
            print(f'Time logger failed on end of event {event_name}. '
                  f'Check that event start time was added')


class Scheduler(ABC):
    @abstractmethod
    def event_due(self, index) -> bool:
        pass


@dataclass
class SimpleScheduler(Scheduler):
    period: int = 24
    offset: int = 0

    def event_due(self, index) -> bool:
        due = False
        if index - self.offset % self.period == 0:
            due = True
        return due


@dataclass
class DTScheduler(Scheduler):
    start_dt: datetime
    interval: timedelta
    custom_events: Tuple[datetime] = tuple([])

    def __post_init__(self):
        self.next_event_due = self.start_dt

    def event_due(self, index: datetime) -> bool:
        due = False
        if self.next_event_due - index <= timedelta(hours=0):
            due = True
            self.next_event_due = index + self.interval
        else:
            for event_dt in self.custom_events:
                if event_dt - index <= timedelta(hours=0):
                    due = True
                    self.custom_events.remove(event_dt)
        return due


@dataclass
class Forecaster(ABC):
    window: timedelta

    @abstractmethod
    def look_ahead(
        self,
        arr: pd.Series,
        start_index: datetime,
        custom_window: timedelta = None
    ):
        pass


@dataclass
class SimpleForecaster(Forecaster):
    window: int

    def look_ahead(
        self,
        arr: pd.Series,
        start_index: int,
        custom_window: int = None
    ):
        window = self.window
        if custom_window:
            window = custom_window
        return arr[start_index: start_index + window]


@dataclass
class DTPerfectForecaster(Forecaster):
    def look_ahead(
        self,
        arr: pd.Series,
        start_index: datetime,
        custom_window: timedelta = None
    ):
        window = self.window
        if custom_window:
            window = custom_window
        fmt = '%Y-%m-%d %H:%M'
        end_time = start_index + window
        return arr[start_index.strftime(fmt): end_time.strftime(fmt)]


class PeakAreas:
    @staticmethod
    def cumulative_peak_areas(sorted_arr):
        diff = np.diff(sorted_arr)
        reverse_index = np.array(range(len(sorted_arr), 1, -1))
        delta_area = diff * reverse_index
        return np.cumsum(np.flip(delta_area))

    @staticmethod
    def peak_area_idx(peak_areas, area):
        return np.searchsorted(peak_areas, area)
