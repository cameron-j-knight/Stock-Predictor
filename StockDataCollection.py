"""
Module used to collect data on market such as close price
open price and volume
Author: Cameron Knight
3/4/17
Copyright 2017, Cameron Knight, All rights reserved.
"""

import yahoo_finance
import datetime
from GoogleTrendsAnalysis import *


def get_stock_name(stock):
    """
    Gets the name of a stock
    :param stock: the stock symbol
    :return: the stocks name
    """

    for i in range(FETCH_ATTEMPTS):
        try:
            share = yahoo_finance.Share(stock)
            return share.get_name()
        except yahoo_finance.YQLResponseMalformedError:
            print("Data not collected. Trying Again")
    return ""


def get_stock_close_list(stock, start_date='FiveYear', end_date='Today'):
    """
    gets the stocks close price in a list from start date to end date
    :param stock: the symbol of the stock to get data for
    :param start_date: the date to start the data collection ('FiveYear' for five years from end date)
    :param end_date: the date to stop the data collection ('Today' to end data collection at today's date)
    :return: a list of stock's close prices from start_date to end_date
    """

    today = datetime.date.today()

    if end_date == 'Today':
        end_date = today
    else:
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    five_years = end_date - datetime.timedelta(LENGTH_YEAR * 5)

    if start_date == 'FiveYear':
        start_date = five_years
    else:
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    for i in range(FETCH_ATTEMPTS):
        try:
            share = yahoo_finance.Share(stock)
            if share.get_name() is not None:
                long_stock = share.get_historical(str(start_date)[:10], str(end_date)[:10])[::-1]
                long_stock = pd.DataFrame(long_stock)
                long_stock = long_stock.set_index(pd.DatetimeIndex(long_stock['Date']))

                del long_stock['Date']
                del long_stock['Symbol']
                long_stock = long_stock.apply(pd.to_numeric, errors='coerce')
                long_stock = long_stock['Close'].values.tolist()
                return long_stock
        except yahoo_finance.YQLResponseMalformedError:
            print("Data not collected. Trying Again")
    return None


def get_market_data(stock, start_date='FiveYear', end_date='Today'):
    """
    gets the market data for a given symbol over a period of time
    :param stock: symbol of stock to search
    :param start_date: the starting date of data collection
    :param end_date:  the ending date of data collection
    :return: pandas data frame of the stock's market data with columbs coresponding to standard market data
    """

    today = datetime.date.today()

    if end_date == 'Today':
        end_date = today
    else:
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    five_years = end_date - datetime.timedelta(LENGTH_YEAR * 5)

    if start_date == 'FiveYear':
        start_date = five_years
    else:
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    for i in range(3):
        try:
            share = yahoo_finance.Share(stock)
            if share.get_name() is not None:
                long_stock = share.get_historical(str(start_date)[:10], str(end_date)[:10])[::-1]
                long_stock = pd.DataFrame(long_stock)
                long_stock = long_stock.set_index(pd.DatetimeIndex(long_stock['Date']))

                del long_stock['Date']
                del long_stock['Symbol']
                long_stock = long_stock.apply(pd.to_numeric, errors='coerce')
                return long_stock
        except yahoo_finance.YQLResponseMalformedError:
            print("Data not collected for " + stock + ". Trying Again: Attempt " + str(i + 1))
    return None


def get_year_data(data):
    data.year_historical = {}
    start = data.dates[0] - datetime.timedelta(LENGTH_YEAR * 2)
    end = data.dates[-1]

    total_close = get_stock_close_list(data.symbol,
                                       str(start)[:USEFUL_TIMESTAMP_CHARS],
                                       str(end)[:USEFUL_TIMESTAMP_CHARS])
    for i in range(len(data.dates)):
        data.year_historical[data.dates[i]] = total_close[i:i + LENGTH_YEAR]
