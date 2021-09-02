import pandas as pd

from utils.data_utils import s3BucketManager
from grid_resources.portfolios import MeritOrderPortfolio
from grid_resources.technologies import Generator, GeneratorTechnoEconomicProperties
from grid_resources.commodities import Fuel, PriceCorrelation, StaticPrice, FuelMarkets, Emissions
from grid_resources.demand import GridDemand
from matplotlib import pyplot as plt

bucket = s3BucketManager('jw-modelling')
folders = ['colombia-portfolio-inputs']

generators_fn = 'generator_economics.json'
demand_fn = 'UnitDemand2013.json'
fuels_fn = 'test_fuels.json'
coal_gas_diesel_data_fn = 'coal_gas_diesel_prices_dollars_per_mwh_monthly.csv'

generator_economics = bucket.s3_json_to_dict(folders, generators_fn)
demand = bucket.s3_json_to_dict(folders, demand_fn)
fuels_data = bucket.s3_json_to_dict(folders, fuels_fn)
coal_gas_diesel_monthly = bucket.s3_csv_to_df(folders, coal_gas_diesel_data_fn)

emissions_tariff = Emissions('carbon_price', 50, '$ / tonne')
interest_rate = 0.03
demand = GridDemand(
    'test',
    'MWh',
    pd.Series(demand['demand'], index=range(len(demand['demand'])))
)

print(fuels_data)
fuels_dict = {}
for fuel in fuels_data['fuels']:
    fuels_dict[fuel['name']] = Fuel(**fuel)


generators = []
for gen, data in generator_economics.items():
    if gen != 'Existing Hydro':
        generators.append(
            Generator(
                gen,
                properties=GeneratorTechnoEconomicProperties.from_dict(
                    gen,
                    data,
                    fuels_dict,
                    emissions_tariff,
                    interest_rate
                ),
            )
        )
print(generators)

print(fuels_dict)

correlated_price_models = PriceCorrelation.from_data(
    coal_gas_diesel_monthly,
    {
        'coal': fuels_dict['coal'],
        'diesel': fuels_dict['diesel']

    },
    'lognormal'
)

static_price_models = StaticPrice({
    'gas': fuels_dict['gas'],
    'biomass': fuels_dict['biomass'],
    'water': fuels_dict['water']
})
market_prices = [static_price_models, correlated_price_models]

fuel_markets = FuelMarkets(
    market_prices
)

portfolio = MeritOrderPortfolio(
    generators,
    [],
    demand
)

portfolio.plot_cost_curves()
portfolio.plot_ldc()
dispatch = portfolio.dispatch()
portfolio.plot_dispatch()
plt.show()
