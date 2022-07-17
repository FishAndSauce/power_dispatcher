from dataclasses import dataclass


@dataclass
class StackPlotConfig:
    plot: bool
    color_map: dict