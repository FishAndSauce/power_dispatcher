import pandas as pd
from matplotlib import pyplot as plt

from utils.data_utils import s3BucketManager
from grid_resources.portfolios import MeritOrderPortfolio, SRMCPortfolio
from grid_resources.technologies import Generator, InstalledGenerator, GeneratorTechnoEconomicProperties
from grid_resources.commodities import Fuel, PriceCorrelation, StaticPrice, Markets, Emissions
from grid_resources.demand import GridDemand



bucket = s3BucketManager('jw-modelling')
folders = ['colombia-portfolio-inputs']

generators_fn = 'generator_economics.json'
unit_demand_fn = 'UnitDemand2013.json'
fuels_fn = 'test_fuels.json'
coal_gas_diesel_data_fn = 'coal_gas_diesel_prices_dollars_per_mwh_monthly.csv'
capacities_fn = 'update.json'

generator_economics = bucket.s3_json_to_dict(folders, generators_fn)
unit_demand = bucket.s3_json_to_dict(folders, unit_demand_fn)
fuels_data = bucket.s3_json_to_dict(folders, fuels_fn)
coal_gas_diesel_monthly = bucket.s3_csv_to_df(folders, coal_gas_diesel_data_fn)
capacities_data = bucket.s3_json_to_dict(folders, capacities_fn)
capacities_df = pd.DataFrame.from_dict(capacities_data).T

# calculate new installation capacities after seed year
seed_year = 2020
capacities_df.index = capacities_df.index.astype(int)
capacities_df = capacities_df[capacities_df.index > 2019]
for gen in capacities_df.columns:
    capacities_df[f'New {gen}'] = capacities_df[gen].diff().cumsum()
    capacities_df.rename(columns={gen: f'Existing {gen}'}, inplace=True)

print(capacities_df)
capacities = capacities_df.loc[2030, :].to_dict()
print(capacities)

emissions_tariff = Emissions('carbon_price', 100, '$ / tonne')
interest_rate = 0.03
unit_demand = GridDemand(
    'test',
    'MWh',
    25 * pd.Series(unit_demand['demand'], index=range(len(unit_demand['demand'])))
)

fuels_dict = {}
for fuel in fuels_data['fuels']:
    fuels_dict[fuel['name']] = Fuel(**fuel)


generators = []
for gen, data in generator_economics.items():
    generators.append(
        InstalledGenerator(
            capacities[gen],
            Generator(
                gen,
                properties=GeneratorTechnoEconomicProperties.from_dict(
                    gen,
                    data,
                    fuels_dict,
                    emissions_tariff,
                    interest_rate
                )
            ),
        )
    )
print(generators)

print(fuels_dict)

static_prices = StaticPrice({
    'biomass': fuels_dict['biomass'],
    'water': fuels_dict['water'],
})

correlated_stochastic_prices = PriceCorrelation.from_data(
    coal_gas_diesel_monthly,
    {
    'coal': fuels_dict['coal'],
    'diesel': fuels_dict['diesel'],
    'gas': fuels_dict['gas'],
    },
    'lognormal'
)

fuel_markets = Markets(
    [static_prices, correlated_stochastic_prices]
)

portfolio = SRMCPortfolio(
    generators,
    [],
    unit_demand
)
#
# portfolio.plot_cost_curves()
# portfolio.plot_ldc()
dispatch = portfolio.dispatch()
portfolio.plot_dispatch()
plt.show()
