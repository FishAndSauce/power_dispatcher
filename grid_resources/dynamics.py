from abc import ABC, abstractmethod
from dataclasses import dataclass

from statistics.stochastics import RandomWindowChoiceModel


@dataclass
class DynamicResource(ABC):
    @abstractmethod
    def refresh(self):
        pass

