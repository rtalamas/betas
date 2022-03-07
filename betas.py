from messari.messari import Messari
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
import matplotlib.ticker as mtick
from datetime import datetime, timedelta
from statsmodels.regression.rolling import RollingOLS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

m = Messari()


def prices(asset_list, start, end):
    """
    Computes asset betas using linear regression approach
    to a given market
    @param asset_list: list
        list of assets
    @param start: ste
        start date
    @param end: str
        end date
    @return: pd.DataFrame
        pandas dataframe without outliers
    """
    price_df = m.get_metric_timeseries(asset_list, 'price', start, end)
    price_df = price_df.xs('close', axis=1, level=1)
    return price_df


def detect_and_remove_outliers(data, threshold=3):
    """
    Computes asset betas using linear regression approach
    to a given market
    @param data: pd.DataFrame
    @param threshold: int
        Outlier threshold
    @return: pd.DataFrame
        pandas dataframe without outliers
    """
    return data[(np.abs(stats.zscore(data)) < threshold).all(axis=1)]


def compute_betas(asset_price_df, market_ticker):
    """
    Computes asset betas using linear regression approach
    to a given market
    @param asset_price_df: pd.DataFrame
        pandas dataframe with asset price data
    @param market_ticker: str
        Market to compute betas against. Data pulled from Yahoo Finance
        if betas are computed against an equity index
    @return: pd.DataFrame
        asset betas pandas dataframe
    """
    betas = {}
    if market_ticker not in asset_price_df.columns:
        start_date = asset_price_df.index[0]
        end_date = asset_price_df.index[-1]
        try:
            market_prices = prices(market_ticker, start_date, end_date)
        except:
            try:
                market_prices = yf.download(tickers=market_ticker, start=start_date.strftime('%Y-%m-%d'),
                                            end=end_date.strftime('%Y-%m-%d'))['Close'].to_frame(name=market_ticker)
            except:
                raise ValueError('Failed to collect price data for selected market')
        asset_price_df = asset_price_df.join(market_prices, how='outer')
    for asset in asset_price_df.columns:
        if asset == market_ticker:
            continue
        else:
            asset_price_tmp = asset_price_df[[market_ticker, asset]].dropna()
            log_returns = np.log(asset_price_tmp / asset_price_tmp.shift(1)).dropna()
            log_returns = detect_and_remove_outliers(log_returns)
            log_x = log_returns[market_ticker]
            log_y = log_returns[asset]
            X = sm.add_constant(log_x, prepend=False)
            y = log_y
            model = sm.OLS(y, X)
            results = model.fit()
            betas[asset] = results.params[market_ticker]
    betas = pd.Series(betas).to_frame(f'{market_ticker}')
    return betas


def betas_dashboard(asset_betas, sector_mapping):
    """
    Generated Betas dashbaord

    Beta is computed by linear regression
    using log asset returns

    @param asset_betas: pd.DataFrame
        pandas dataframe with asset betas
    @param sector_mapping: dict
        Dictionary of asset: sector mapping
    @return: matplotlib fig
        figure containing Hot Ball of Money Monitor
    """
    # Construct color map from sector mapping
    sectors = list(set(sector_mapping.values()))
    cmap = cm.get_cmap('Spectral', len(sectors))
    cmap = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    color_mapping = {sector: color for sector, color in zip(sectors, cmap)}

    asset_betas['sector'] = asset_betas.index.map(sector_mapping)
    color_list = [color_mapping[i] for i in asset_betas['sector']]
    fig = plt.figure(figsize=(30, 15))
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[sector]) for sector in sectors]
    for col, i in zip(asset_betas.columns, range(1, len(asset_betas.columns))):
        if col == 'sector':
            continue
        else:
            ax = asset_betas[col].plot(kind='bar', ax=plt.subplot(int(str('23') + str(i))), color=color_list)
            ax.set_title(f'Beta to {col}', fontsize=18)
            ax.set_ylabel('Beta')
            ax.axhline(y=1, color='r', linestyle='--')
            ax.axhline(y=1.5, color='r', linestyle='--')
            ax.axhline(y=2, color='r', linestyle='--')
            plt.suptitle(f'Asset Betas by Market', size=40)
            plt.figlegend(handles, sectors, loc='lower center', ncol=3, prop={'size': 16})
    average_sector_betas = asset_betas.groupby('sector').mean()
    average_sector_betas = average_sector_betas.reindex(list(asset_betas['sector'].unique()))
    color_list_average_values = [color_mapping[i] for i in average_sector_betas.index]
    average_sector_betas.index = [x.replace(' ', '\n') for x in average_sector_betas.index]
    for col, i in zip(average_sector_betas.columns, range(4, len(average_sector_betas.columns) + 4)):
        tmp_df = average_sector_betas[col].to_frame(name=col)
        ax = tmp_df[col].plot(kind='bar', ax=plt.subplot(int(str('23') + str(i))), rot=0, legend=False,
                              color=color_list_average_values)
        split_col = col.split(' ')
        ax.set_title(f'Average Sector Beta to {col}', fontsize=18)
        ax.set_ylabel('Average Sector Beta')
        ax.set_xlabel('')
        ax.axhline(y=1, color='r', linestyle='--')
    fig.savefig('betas.png', dpi=300)


def sector_to_asset_mapping(sector_mapping_dict):
    """
    Generates a dictionary of using sector as key and
    a list of assets as value
    """
    sector_to_asset_dict = {}
    for s in set(list(sector_mapping_dict.values())):
        sector_grouping = []
        for key, value in sector_mapping_dict.items():
            if value == s:
                sector_grouping.append(key)
            else:
                continue
        sector_to_asset_dict[s] = sector_grouping
    return sector_to_asset_dict


def rolling_sector_beta(asset_price_df, market_ticker, sector_mapping, rolling_window):
    """
    Computes rolling sector beta to a given market.
    Betas are computed by linear regression
    using log asset returns

    @param asset_price_df: pd.DataFrame
        pandas dataframe with asset prices
    @param market_ticker: str
        Market to compute betas against. Data pulled from Yahoo Finance
        if betas are computed against an equity index
    @param sector_mapping: dict
        Dictionary of asset: sector mapping
    @param rolling_window: int
        Rolling window for beta calculation
    @return: pd.DataFrame
        rolling asset betas pandas dataframe
    """
    rolling_betas = pd.DataFrame(index=asset_price_df.index)
    if market_ticker not in asset_price_df.columns:
        print(f'Collecting data for {market_ticker}...')
        start_date = asset_price_df.index[0]
        end_date = asset_price_df.index[-1]
        try:
            market_prices = prices(market_ticker, start_date, end_date)
        except:
            try:
                market_prices = yf.download(tickers=market_ticker, start=start_date.strftime('%Y-%m-%d'),
                                            end=end_date.strftime('%Y-%m-%d'))['Close'].to_frame(name=market_ticker)
            except:
                raise ValueError('Failed to collect price data for selected market')
        asset_price_df = asset_price_df.join(market_prices, how='outer')

    for asset in asset_price_df.columns:
        if asset == market_ticker:
            continue
        else:
            asset_price_tmp = asset_price_df[[market_ticker, asset]].dropna()
            log_returns = np.log(asset_price_tmp / asset_price_tmp.shift(1)).dropna()
            log_returns = detect_and_remove_outliers(log_returns)
            log_x = log_returns[market_ticker]
            log_y = log_returns[asset]
            model = RollingOLS(endog=log_y, exog=log_x, window=rolling_window)
            rres = model.fit()
            betas = rres.params
            betas.columns = [asset]
            rolling_betas = rolling_betas.join(betas, how='outer')

    # Compute sector betas as the average beta of each asset within a sector
    sector_to_asset_list = sector_to_asset_mapping(sector_mapping)

    rolling_sector_betas = pd.DataFrame(index=asset_price_df.index)
    for sector, list_of_assets in sector_to_asset_list.items():
        if market_ticker in list_of_assets:
            list_of_assets.remove(market_ticker)
        sector_beta = rolling_betas[list_of_assets].mean(axis=1).to_frame(name=f'{sector} ({len(list_of_assets)})')
        rolling_sector_betas = rolling_sector_betas.join(sector_beta, how='outer')

    # linear interpolation when missing values
    rolling_sector_betas.interpolate('linear', inplace=True)
    return rolling_sector_betas
