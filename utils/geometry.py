import json
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Coord:
    x: float
    y: float
    name: str = None

    """ Coordinate representing a position/point in 2D space
    Attributes:
        x (float): Value of x at position
        y (float): Value of y at position
    """

    def find_gradient_between(self, other_coord):
        rise = other_coord.y - self.y
        run = other_coord.x - self.x
        if run == 0.0:
            return None
        else:
            return rise / run


@dataclass
class CoordCollection(object):
    coords: List[Coord]

    @property
    def x_list(self):
        return list([p.x for p in self.coords])

    @property
    def y_list(self):
        return list(p.y for p in self.coords)

    @property
    def name_list(self):
        return list(p.name for p in self.coords)

    @property
    def coords_list(self):
        return list((p.x, p.y) for p in self.coords)

    @property
    def coords_dict(self):
        return {coord.name: coord for coord in self.coords}

    def max_x(self):
       return self.x_list.index(max(self.x_list))

    def min_x(self):
        return self.x_list.index(max(self.x_list))

    def find_envelope(self):
        pass

@dataclass
class Line:
    """Line in the form y = mx + b:
    """
    gradient: float
    y_intercept: float

    def find_y_at_x(self, x):
        """ Finds the value of y for a given value of x

        Args:
            x (float): Nominal x value
        Returns:
            float: Value of y at nominal x value
        """
        # y = mx + b
        return x * self.gradient + self.y_intercept

    def find_x_at_y(self, y):
        """ Finds the value of x for a given value of y

        Args:
            y (float): Nominal y value
        Returns:
            float: Value of x at nominal x value
        """
        # rearange y = mx + b to find x
        return (y - self.y_intercept) / self.gradient

    def find_intercept_on_line(self, other_line):
        ''' Finds the coordinates of intecept of self and another Line
            (if one exists)
        
        Args:
            other_line (Line): Other Line object which may intersect with self
        
        Returns:
            Coord: Coord object representing intercept x and y values
        '''

        m_self = self.gradient
        m_other = other_line.gradient

        # Test for parallel lines
        if m_self != m_other:
            # Find intercept two lines in the form y = mx + b
            # Just high school algebra...
            b_self = self.y_intercept
            b_other = other_line.y_intercept
            x = (b_self - b_other) / (m_other - m_self)
            intercept = Coord(x, self.find_y_at_x(x))
        else:
            intercept = Coord(None, None)  # no intercept for parallel lines
        return intercept


    # def set_plot(self, x_bounds, **kwargs):
    #
    #     x_vals = x_bounds
    #
    #     y_val_1 = self.find_y_at_x(x_bounds[0])
    #     y_val_2 = self.find_y_at_x(x_bounds[1])
    #     y_vals = [y_val_1, y_val_2]
    #
    #     return XYline(x_vals, y_vals, label=self.equation_string, **kwargs)