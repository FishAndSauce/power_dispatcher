import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Type, List, Dict
from scipy.stats import norm
from scipy.linalg import cholesky
from abc import ABC, abstractmethod

supported_distributions = ['normal']
supported_correlation_distributions = ['normal', 'lognormal']


@dataclass
class Validator:
    @staticmethod
    def square_matrix(matrix, attr_name):
        if not matrix.shape[0] == matrix.shape[1]:
            raise ValueError(f'Invalid matrix shape ({matrix.shape}): '
                             f'{attr_name} must be a square matrix (mxm)')

    @staticmethod
    def matching_length(a, b, a_name, b_name):
        if not len(a) == len(b):
            raise ValueError(f'Invalid length ({len(a)}): '
                            f'{a_name} length must match length of {b_name} ({len(b)})')

    @staticmethod
    def options(value, options, attr_name):
        if value not in options:
            raise ValueError(f'Invalid choice: {attr_name} must be one of the following: {", ".join(options)}')

    @staticmethod
    def multivariate_data(data):
        if data.shape[0] < 2:
            raise ValueError(f'Invalid data: data must have 2 or more columns to be multivariate')


class DistributionModel(ABC):
    @staticmethod
    @abstractmethod
    def generate_samples(mean, std_dev, n):
        pass


class NormalDistribution(DistributionModel):
    @staticmethod
    def generate_samples(mean, std_dev, n):
        samples = np.random.normal(
            mean,
            std_dev,
            n
        )
        if n == 1:
            return samples[0]
        else:
            return samples


class StochasticModel(ABC):
    @abstractmethod
    def generate_samples(self, number_samples=1):
        pass


@dataclass
class RandomWindowChoiceModel(StochasticModel):
    data: np.ndarray

    @property
    def last_idx(self):
        return len(self.data) - 1

    def generate_samples(self, number_samples=1):
        start_index = np.random.randint(
            0,
            self.last_idx - number_samples
        )
        end_index = start_index + number_samples
        return self.data[start_index: end_index]


@dataclass
class RandomArrayChoiceModel(StochasticModel):
    data: List[list]

    def generate_samples(self, number_samples=1) -> np.ndarray:
        random_idx = np.random.randint(
            0,
            len(self.data),
            size=number_samples
        )
        samples = []
        for idx in random_idx:
            samples.append(self.data[idx])
        if len(samples) > 1:
            return np.array(samples)
        else:
            return np.array(samples[0])


@dataclass
class ComplementaryRandomArrayChoiceModel(StochasticModel):
    data: Dict[str, List[list]]

    def generate_samples(self, number_samples=1) -> Dict[str, np.ndarray]:
        random_idx = np.random.randint(
            0,
            len(self.data),
            size=number_samples
        )
        sample_dict = {}
        for name, data in self.data.items():
            samples = []
            for idx in random_idx:
                samples.append(data[idx])
            if len(samples) > 1:
                sample_dict[name] = np.array(samples)
            else:
                sample_dict[name] = np.array(samples[0])
        return sample_dict


@dataclass
class DistributionModel(StochasticModel):
    mean: float
    std_dev: float
    distribution: Type[DistributionModel]

    def generate_samples(self, number_samples=1):
        return self.distribution.generate_samples(
            self.mean,
            self.std_dev,
            number_samples
        )


@dataclass
class CorrelatedDistributionModel(StochasticModel):
    data_names: List[str]
    norm_covariance: np.ndarray
    distribution_means: np.ndarray
    distribution_std: np.ndarray
    distribution_type: str

    def __post_init__(self):
        validator = Validator()
        validator.square_matrix(
            self.norm_covariance,
            'covariance_matrix'
        )
        validator.matching_length(
            self.distribution_means,
            self.norm_covariance,
            'distribution_means',
            'covariance_matrix'
        )
        validator.matching_length(
            self.data_names,
            self.distribution_means,
            'data_names',
            'distribution_means'
        )
        validator.options(
            self.distribution_type,
            supported_correlation_distributions,
            'distribution',
        )

    @staticmethod
    def from_data(data: pd.DataFrame, distribution: str):
        Validator().multivariate_data(data)
        Validator().options(
            distribution,
            supported_correlation_distributions,
            'distribution'
        )
        if distribution == 'normal':
            variables_vectors = list([data[data.columns[i]] for i in range(len(data.columns))])
            covariance_matrix = np.cov(variables_vectors)
            distribution_means = data.mean().values
            distribution_std = data.std().values
            return CorrelatedDistributionModel(
                data.columns.to_list(),
                covariance_matrix,
                distribution_means,
                distribution_std,
                distribution
            )
        if distribution == 'lognormal':
            data_logged = np.log(data)
            variables_vectors = list([data_logged[data_logged.columns[i]] for i in range(len(data.columns))])
            covariance_matrix = np.cov(variables_vectors)
            distribution_means = data.mean().values
            distribution_std = data.std().values
            return CorrelatedDistributionModel(
                data.columns.to_list(),
                covariance_matrix,
                distribution_means,
                distribution_std,
                distribution
            )

    def standard_normal_samples(self, number_samples=1):
        return norm.rvs(
            scale=1,
            size=(len(self.norm_covariance), number_samples),
        )

    def correlated_normal_samples(self, number_samples=1):
        normal_sample = self.standard_normal_samples(number_samples)
        cky_decomp = cholesky(self.norm_covariance)
        correlated_normal_sample = np.dot(cky_decomp, normal_sample)
        return correlated_normal_sample

    def correlated_lognormal_samples(self, number_samples=1):
        correlated_normal_sample = self.correlated_normal_samples(number_samples)
        lognormal_sample = np.exp(correlated_normal_sample)
        scaled_lognormal_sample = np.multiply(
            self.distribution_means,
            lognormal_sample.transpose()
        ).transpose()
        return scaled_lognormal_sample

    def generate_samples(self, number_samples=1) -> pd.Series:
        if self.distribution_type == 'normal':
            correlated_normal_samples = self.correlated_normal_samples(number_samples)
            mean_cols = list(
                [[self.distribution_means[i]] * number_samples for i in range(len(self.distribution_means))]
            )
            scaled_normal_sample = correlated_normal_samples + mean_cols
            return pd.Series(
                scaled_normal_sample.flatten(),
                index=self.data_names
            )
        if self.distribution_type == 'lognormal':
            return pd.Series(
                self.correlated_lognormal_samples(number_samples).flatten(),
                index=self.data_names
            )


@dataclass
class StochasticResource(ABC):
    @abstractmethod
    def refresh(self):
        pass