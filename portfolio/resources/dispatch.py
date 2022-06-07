from dataclasses import dataclass, field
from typing import Iterable, Union, Optional

import numpy as np


@dataclass
class DispatchVector:
    """ Specification of charge/discharge - both as vectors of positive floats

    Net charge is represented as single float where positive indicates discharge

    Charge and discharge are enforced as mutually exclusive - (I.e. Discharge should
    never occur at the same index as charge)
    """
    name: str
    charge: Optional[np.ndarray] = field(default=None)
    discharge: Optional[np.ndarray] = field(default=None)
    excess: Optional[np.ndarray] = field(default=None)

    @property
    def vector_list(self) -> list:
        return [
            self.charge,
            self.discharge,
            self.excess,
        ]

    @property
    def vector_lengths(self) -> list:
        return list([len(v) if v is not None else 0 for v in self.vector_list])

    @property
    def as_net(self):
        """ Representation of dispatch as vector of net values - positive value indicates
        discharge, negative indicates charge
        """
        return self.discharge - self.charge

    def fill_zeros(self):
        length = self.validate_equal_lengths()
        self.charge = np.array(length * [0]) if self.charge is None else self.charge
        self.discharge = np.array(length * [0]) if self.discharge is None else self.discharge
        self.excess = np.array(length * [0]) if self.excess is None else self.excess

    def __post_init__(self):
        self.fill_zeros()
        self.validate_positive_vector_values()
        self.validate_charge_discharge_mutual_exclusion()

    def validate_equal_lengths(self) -> int:
        non_zero_lengths = [l for l in self.vector_lengths if l]
        if len(set(non_zero_lengths)) > 1:
            raise ValueError(
                f'Arrays for charge, discharge and excess must be equal length. '
                f'Respective lengths received: {self.vector_lengths}'
            )
        return max(self.vector_lengths)

    def validate_positive_vector_values(self):
        message = 'Vectors must be >= 0.0. The {} vector has negative values at the following indices {}'.format
        charge_neg_indices = np.where(self.charge < 0.0)[0]
        discharge_neg_indices = np.where( self.discharge < 0.0)[0]
        excess_neg_indices = np.where(self.excess < 0.0)[0]

        if charge_neg_indices.size > 0:
            raise ValueError(message('charge', charge_neg_indices))
        if discharge_neg_indices.size > 0:
            raise ValueError(message('discharge', discharge_neg_indices))
        if excess_neg_indices.size > 0:
            raise ValueError(message('excess', excess_neg_indices))

    def validate_charge_discharge_mutual_exclusion(self):
        charge_active = self.charge > 0.0
        discharge_active = self.discharge > 0.0
        both_active = charge_active * discharge_active
        if any(both_active):
            indices = np.where(both_active)
            raise ValueError(
                f'Dispatch charge and discharge are mutually exclusive - they must not be active'
                f' (i.e. positive and non-zero) at same indices. '
                f'Charge and discharge are active at the following indices: {indices[0]}'
            )

    @classmethod
    def from_raw_floats(cls, name, dispatch_vector: Union[np.ndarray, list]):
        """ Create instance from raw float where positive value
        is interpreted as discharge and negative dispatch interpreted as charge
        """
        return cls(
            name=name,
            charge=np.where(dispatch_vector < 0.0, -dispatch_vector, 0.0),
            discharge=np.where(dispatch_vector > 0.0, dispatch_vector, 0.0),
        )