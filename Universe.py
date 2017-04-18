"""
Author: Cameron Knight
Copyright 2017, Cameron Knight, All rights reserved.
"""

import datetime
from Util import *
import StockDataCollection
import numpy as np
import matplotlib.pyplot as plt
import random

# TODO incorporate trade costs
# TODO add a util method to handle and standardize any datetime format (text, pandas, and datetime) and apply to funcs


class Order:
    """
    an order for the number of stocks to own after a given execution of an algorithm
    """
    def __init__(self):
        """
        creates a stock order
        """
        self.orders = {}

    def add_order_from_dict(self, stock_amounts):
        """
        orders a stock using a dictionary of stocks and ammounts
        :param stock_amounts: dictionary in form (symbol, amount to hold)
        :return: None
        """
        for stock in stock_amounts.keys():
            self.orders[stock] = stock_amounts[stock]

    def buy(self, stock, amount):
        """
        buys an amount of a stock
        :param stock: symbol of stock to buy
        :param amount: number of stocks to buy
        :return: None
        """
        self.orders[stock] += amount

    def sell(self, stock, amount):
        """
        sells a given stock (negative amount is a short)
        :param stock: symbol of stock to buy
        :param amount: the number of stocks to sell
        :return: None
        """
        self.orders[stock] -= amount

    def order(self, stock, amount):
        """
        orders a number of stocks so you are holding a given amount (negative is short)
        :param stock: symbol of stock to order
        :param amount: number of stock you want to hold
        :return: None
        """
        self.orders[stock] = amount

    def __str__(self):
        return str(self.orders)

    # TODO add limit buys and sells to be more interactive with the market


class StockData(object):
    """
    holds the data for a stock
    """
    def __init__(self, stock_name=None, stock_symbol=None):
        self.name = stock_name
        self.symbol = stock_symbol
        self.dates = None
        self.str_dates = None
        self.day_historical = None
        self.week_historical = None
        self.month_historical = None
        self.year_historical = None
        self.market = None
        self.related_terms = []
        self.google_trends = None
        self.ad_meter = None
        self.twitter_feed = None
        self.twitter_sentiment = None
        self.reddit_feed = None
        self.reddit_sentiment = None
        self.press_release_feed = None
        self.press_release_sentiment = None
        self.moving_average_200 = None
        self.moving_average_50 = None
        self.trailing_moving_average_12 = None
        self.trailing_moving_average_26 = None
        self.volatility = None
        self.relative_strength_index = None
        self.moving_average_convergence_divergence = None
        self.golden_cross = None
        self.pullback = None
        self.on_balance_volume = None
        self.pivot_point = None
        self.advance_decline_line = None
        self.average_directional_index = None
        self.aroon = None
        self.stochastic_oscillator = None
        self.layoff_analysis = None
        self.auto_encoded_stock_info = None
        self.stock_twit = None
        self.stock_twit_sentiment = None
        self.day_growth = None
        self.week_growth = None
        self.month_growth = None
        self.day_futures = None
        self.week_futures = None
        self.month_futures = None

        self.position = None

        self._verbose = False

    def to_stock_dataframe_day(self, date):
        """
        gets the stock dataframe for a single day
        :param date: day to get data for
        :return: dictionary of features and values for the given day
        """
        if type(date) is not datetime.datetime and type(date) is not pd.tslib.Timestamp:
            date = datetime.datetime.strptime(date, "%Y-%m-%d")

        class_data = [i for i in dir(self) if not callable(getattr(self, i)) and
                        not i.startswith("__") and type(getattr(self, i)) is pd.DataFrame]
        df = pd.DataFrame()
        for i in class_data:
            df = join_features(df, getattr(self, i), fill_method=FillMethod.FUTURE_KROGH)
        return df.ix[date, :]

    def to_stock_dataframe_range(self, start_date=None, end_date=None):
        """
        gets stock dataframe for a range of dates
        :param start_date: date to start collection None for earliest
        :param end_date: date to end data collection None for last
        :return: data frame with data in the range for the stock
        """
        if end_date is None:
            end_date = self.dates[-2]
        if type(end_date) is pd.tslib.Timestamp:
            end_date = end_date.strftime("%Y-%m-%d")
        if type(end_date) is not datetime.datetime and type(end_date) is not pd.tslib.Timestamp:
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        try:
            end_date = self.dates[list(self.dates).index(end_date) + 1]
        except:
            end_date = "Last"
        if start_date is None:
            start_date = self.dates[0]

        if type(start_date) is not datetime.datetime and type(start_date) is not pd.tslib.Timestamp:
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

        class_data = [i for i in dir(self) if not callable(getattr(self, i)) and not i.startswith("__") and
                      type(getattr(self, i)) is pd.DataFrame]
        df = pd.DataFrame()
        for i in class_data:
            df = join_features(df, getattr(self, i), fill_method=FillMethod.FUTURE_KROGH)
        if end_date is "Last":
            print(df.ix[start_date:, :])
            return df.ix[start_date:, :]
        return df.ix[start_date:end_date, :]

    def to_stock_data_day(self, date):
        """
        gets the stock data for a single day
        :param date: day to get data for
        :return: dictionary of features and values for the given day
        """
        if type(date) is not datetime.datetime and type(date) is not pd.tslib.Timestamp:
            date = datetime.datetime.strptime(date, "%Y-%m-%d")

        dataframes = [i for i in dir(self) if not callable(getattr(self, i)) and not i.startswith("__")
                      and type(getattr(self, i)) is pd.DataFrame]
        dictionaries = [i for i in dir(self) if not callable(getattr(self, i)) and not i.startswith("__")
                      and type(getattr(self, i)) is dict]
        constant_values = [i for i in dir(self) if not callable(getattr(self, i)) and not i.startswith("__")
                      and getattr(self, i) is not None and i not in dataframes and i not in dictionaries]
        new_stock_data = StockData()

        for i in dataframes + dictionaries:
            setattr(new_stock_data, i, getattr(self, i)[date])

        for i in constant_values:
            setattr(new_stock_data, i, getattr(self, i))

        new_stock_data.dates = [date]
        new_stock_data.str_dates = [str(date)[:USEFUL_TIMESTAMP_CHARS]]

        return new_stock_data

    def to_stock_data_range(self, start_date=None, end_date=None):
        """
        gets stock data for a range of dates
        :param start_date: date to start collection None for earliest
        :param end_date: date to end data collection None for last
        :return: data frame with data in the range for the stock
        """
        # standardize dates
        if end_date is None:
            end_date = self.dates[-2]
        if type(end_date) is pd.tslib.Timestamp:
            end_date = end_date.strftime("%Y-%m-%d")
        if type(end_date) is not datetime.datetime and type(end_date) is not pd.tslib.Timestamp:
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        try:
            end_date = self.dates[list(self.dates).index(end_date) + 1]
        except:
            end_date = "Last"

        if start_date is None:
            start_date = self.dates[0]
        if type(start_date) is not datetime.datetime and type(start_date) is not pd.tslib.Timestamp:
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

        if end_date is "Last":
            dates = list(self.dates)[list(self.dates).index(start_date):]
        else:
            dates = list(self.dates)[list(self.dates).index(start_date):list(self.dates).index(end_date)]

        # find functions to set
        dataframes = [i for i in dir(self) if not callable(getattr(self, i)) and not i.startswith("__")
                      and type(getattr(self, i)) is pd.DataFrame]
        dictionaries = [i for i in dir(self) if not callable(getattr(self, i)) and not i.startswith("__")
                        and type(getattr(self, i)) is dict]
        constant_values = [i for i in dir(self) if not callable(getattr(self, i)) and not i.startswith("__")
                           and getattr(self, i) is not None and i not in dataframes and i not in dictionaries]

        # transfer new data
        new_stock_data = StockData()

        for i in constant_values:
            setattr(new_stock_data, i, getattr(self, i))

        for i in dataframes:
            if end_date is not "Last":
                setattr(new_stock_data, i, getattr(self, i).ix[start_date:end_date])
            else:
                setattr(new_stock_data, i, getattr(self, i).ix[start_date:])

        for i in dictionaries:
            new_dict = {}
            for d in dates:
                new_dict[d] = getattr(self, i)[d]
            setattr(new_stock_data, i, new_dict)

        new_stock_data.dates = dates
        new_stock_data.str_dates = [str(d)[:USEFUL_TIMESTAMP_CHARS] for d in dates]

        return new_stock_data

    def __str__(self):
        def convert_list_to_string(lst):
            converted_string = ""
            if len(lst) > 5:
                return convert_list_to_string(lst[:2]) + " ... " + convert_list_to_string(lst[-2:])
            for i in lst:
                converted_string += "{0:.2f}".format(i + 0.0001) + ", "
            return converted_string[:-2]

        def build_list_string(info, name="This"):
            st = ""
            if info is not None:
                st += name + ": " + convert_list_to_string(info[self.dates[0]])
                st += "\n\t"
                st += "                      ....    ....    ...."
                st += "\n\t"
                st += "                 " + convert_list_to_string(info[self.dates[-1]])
                st += "\n\t"
            else:
                st += name + ": " + "Not added"
                st += "\n\t"
            return st

        def build_scalar_string(info, name="This"):
            st = ""
            if info is not None:
                st += name + ": " + convert_list_to_string(
                    list(info.ix[:, 0]))
                st += "\n\t"
            else:
                st += name + ": " + "not added"
                st += "\n\t"
            return st

        build_string = self.symbol + ", " + self.name
        build_string += ", from: " + self.str_dates[0] + " to: " + self.str_dates[-1] + "\n\t"

        build_string += build_list_string(self.week_historical, "Week historical")
        build_string += build_list_string(self.month_historical, "Month historical")
        build_string += build_list_string(self.year_historical, "Year historical")

        build_string += build_scalar_string(self.moving_average_200, "200 Day Moving Average")
        build_string += build_scalar_string(self.moving_average_50, "50 Day Moving Average")

        build_string += build_scalar_string(self.trailing_moving_average_12, "12 Day Trailing Moving Average")
        build_string += build_scalar_string(self.trailing_moving_average_26, "26 Day Trailing Moving Average")

        build_string += build_scalar_string(self.volatility, "Volatility")

        build_string += build_scalar_string(self.relative_strength_index, "Relative Strength Index")

        build_string += build_scalar_string(self.moving_average_convergence_divergence,
                                            "Moving Average Covergance Divergance")

        build_string += build_scalar_string(self.on_balance_volume, "On Balance Volume")
        build_string += build_scalar_string(self.aroon, "Aroon Oscillator")
        build_string += build_scalar_string(self.golden_cross, "Golden Cross")
        build_string += build_list_string(self.month_futures,  "Month Futures")
        build_string += build_list_string(self.week_futures, "Week Futures")
        build_string += build_list_string(self.day_futures, "Day Futures")

        build_string += build_scalar_string(self.month_growth, "Month Growth")
        build_string += build_scalar_string(self.week_growth, "Week Growth")
        build_string += build_scalar_string(self.day_growth, "Day Growth")

        build_string += build_scalar_string(self.position, "Position")

        return build_string[:-2]


class DataTypes:
    """
    Available types of data to collect for a stock
    """

    def google_trends(data, historical=True):
        verbose_message("Collecting Google Trend Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def ad_meter(data, historical=True):
        verbose_message("Collecting AD meter Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def year_data(data, historical=True):
        verbose_message("Collecting Year Data for " + data.symbol)
        if not historical:
            pass
        else:
            StockDataCollection.get_year_data(data)

    def month_data(data, historical=True):
        verbose_message("Collecting Month Data for " + data.symbol)
        if not historical:
            pass
        else:
            data.month_historical = {}
            if data.year_historical is None:
                verbose_message(
                    "\tYear historical data required for month historical data: Collecting year historical data")
                StockDataCollection.get_year_data(data)
            for d in data.dates:
                data.month_historical[d] = data.year_historical[d][len(data.year_historical[d]) - LENGTH_MONTH:]

    def week_data(data, historical=True):
        verbose_message("Collecting Week Data for " + data.symbol)
        if not historical:
            pass
        else:
            data.week_historical = {}
            if data.year_historical is None:
                verbose_message(
                    "\tYear historical data required for week historical data: Collecting year historical data")
                StockDataCollection.get_year_data(data)
            for d in data.dates:
                data.week_historical[d] = data.year_historical[d][len(data.year_historical[d]) - LENGTH_WEEK:]

    def day_data(data, historical=True):
        verbose_message("Collecting Day Data for " + data.symbol)
        if not historical:
            pass
        else:
            data.day_historical = {}
            if data.year_historical is None:
                verbose_message(
                    "\tYear historical data required for day historical data: Collecting year historical data")
                StockDataCollection.get_year_data(data)
            for d in data.dates:
                data.day_historical[d] = data.year_historical[d][len(data.year_historical[d]) - LENGTH_DAY:]

    def twitter_feed(data, historical=True):
        verbose_message("Collecting Twitter Feed Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def twitter_sentiment(data, historical=True):
        verbose_message("Collecting Twitter Sentiment Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def reddit_feed(data, historical=True):
        verbose_message("Collecting Reddit Feed Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def reddit_sentiment(data, historical=True):
        verbose_message("Collecting Reddit Sentiment Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def press_release_feed(data, historical=True):
        verbose_message("Collecting Press Release Feed Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def press_release_sentiment(data, historical=True):
        verbose_message("Collecting Press Release Sentiment Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def moving_average_200(data, historical=True):
        verbose_message("Collecting 200 day moving average Data for " + data.symbol)
        if not historical:
            pass
        else:
            data.moving_average_200 = []
            if data.year_historical is None:
                verbose_message(
                    "\tYear historical data required for 200 day moving average data: Collecting year historical data")
                StockDataCollection.get_year_data(data)
            for d in data.dates:
                data.moving_average_200 += \
                    [np.mean(data.year_historical[d][LENGTH_YEAR - 200:])]
            data.moving_average_200 = pd.DataFrame({"MA_200": data.moving_average_200}).set_index(data.dates)

    def moving_average_50(data, historical=True):
        verbose_message("Collecting 50 day moving average Data for " + data.symbol)
        if not historical:
            pass
        else:
            data.moving_average_50 = []
            if data.year_historical is None:
                verbose_message(
                    "\tYear historical data required for 50 day moving average data: Collecting year historical data")
                StockDataCollection.get_year_data(data)
            for d in data.dates:
                data.moving_average_50 += \
                    [np.mean(data.year_historical[d][LENGTH_YEAR - 50:])]
            data.moving_average_50 = pd.DataFrame({"MA_50": data.moving_average_50}).set_index(data.dates)

    def trailing_moving_average_12(data, historical=True):
        verbose_message("Collecting 12 day trailing moving average Data for " + data.symbol)
        if not historical:
            pass
        else:
            data.trailing_moving_average_12 = []
            if data.year_historical is None:
                verbose_message(
                    "\tYear historical data required for 12 day trailing moving average data: \
                    Collecting year historical data")
                StockDataCollection.get_year_data(data)
            last_ema = None
            for d in data.dates:
                time_period = 12
                if last_ema is None:
                    last_ema = np.mean(data.year_historical[d][LENGTH_YEAR - 12:])
                else:
                    current_close = data.year_historical[d][-1]
                    last_ema = current_close * (2/(time_period + 1)) + last_ema*(1-(2/(time_period + 1)))
                data.trailing_moving_average_12 += [last_ema]
            data.trailing_moving_average_12 = pd.DataFrame({"EMA_12":
                                                                data.trailing_moving_average_12}).set_index(data.dates)

    def trailing_moving_average_26(data, historical=True):
        verbose_message("Collecting 26 day trailing moving average Data for " + data.symbol)
        if not historical:
            pass
        else:
            data.trailing_moving_average_26 = []
            if data.year_historical is None:
                verbose_message(
                    "\tYear historical data required for 26 day trailing moving \
                    average data: Collecting year historical data")

                StockDataCollection.get_year_data(data)
            last_ema = None
            for d in data.dates:
                time_period = 26
                if last_ema is None:
                    last_ema = np.mean(data.year_historical[d][LENGTH_YEAR - 26:])
                else:
                    current_close = data.year_historical[d][-1]
                    last_ema = current_close * (2 / (time_period + 1)) + last_ema * (1 - (2 / (time_period + 1)))
                data.trailing_moving_average_26 += [last_ema]
            data.trailing_moving_average_26 = pd.DataFrame({"EMA_26":
                                                                data.trailing_moving_average_26}).set_index(data.dates)

    def volatility(data, historical=True):
        verbose_message("Collecting volatility Data for " + data.symbol)
        if not historical:
            pass
        else:
            data.volitility = []
            if data.year_historical is None:
                verbose_message(
                    "\tYear historical data required for 50 day moving average data: Collecting year historical data")
                StockDataCollection.get_year_data(data)
            for d in data.dates:
                data.volitility += [float(np.var(data.year_historical[d]))]

            data.volatility = pd.DataFrame({"Volatility": data.volitility}).set_index(data.dates)

    def relative_strength_index(data, historical=True, smooth=True):
        verbose_message("Collecting rsi Data for " + data.symbol)
        if not historical:
            pass
        else:
            data.relative_strength_index = []
            if data.year_historical is None:
                verbose_message(
                    "\tYear historical data required for RSI data: Collecting year historical data")
                StockDataCollection.get_year_data(data)
            prev_gain_loss = None
            for d in data.dates:
                days_14 = list(data.year_historical[d][LENGTH_YEAR - 14 - 1:])
                days_14_gains = [b-a for a,b in zip(days_14, days_14[1:])]
                gains = []
                losses = []
                for day in days_14_gains:
                    if day > 0:
                        gains += [day]
                    else:
                        losses += [-day]
                gain = 0 if len(gains) == 0 else float(np.mean(gains))
                loss = 0 if len(losses) == 0 else float(np.mean(losses))
                if smooth:
                    if prev_gain_loss is not None:
                        gain = (prev_gain_loss[0] * 13 + gain) / 14
                        loss = (prev_gain_loss[1] * 13 + loss) / 14

                    prev_gain_loss = (gain,loss)

                relative_strength = 0 if loss == 0 else gain/loss
                rsi = 100 - (100/(1 + relative_strength))
                data.relative_strength_index += [rsi]

            data.relative_strength_index = pd.DataFrame({"RSI": data.relative_strength_index}).set_index(data.dates)

    def moving_average_convergence_divergence(data, historical=True):
        verbose_message("Collecting macd Data for " + data.symbol)
        if not historical:
            pass
        else:
            data.moving_average_convergence_divergence = []

            if data.trailing_moving_average_26 is None:
                verbose_message(
                    "\t26 day trailing moving average data required for moving average convergence divergence \
                    data: Collecting 26 day trailing moving average data")

                DataTypes.trailing_moving_average_26(data)

            if data.trailing_moving_average_12 is None:
                verbose_message(
                    "\t16 day trailing moving average data required for moving average convergence divergence \
                    data: Collecting 16 day trailing moving average data")

                DataTypes.trailing_moving_average_12(data)

            for d in data.dates:
                data.moving_average_convergence_divergence += [data.trailing_moving_average_12.get_value(d,'EMA_12') -
                                                               data.trailing_moving_average_26.get_value(d,'EMA_26')]

            data.moving_average_convergence_divergence = \
                pd.DataFrame({"MACD": data.moving_average_convergence_divergence}).set_index(data.dates)

    def golden_cross(data, historical=True):
        verbose_message("Collecting golden cross Data for " + data.symbol)
        if not historical:
            pass
        else:
            if data.moving_average_50 is None:
                verbose_message(
                    "\t50 day moving average data required for golden cross \
                    data: Collecting 50 day  moving average data")

                DataTypes.moving_average_50(data)

            if data.moving_average_200 is None:
                verbose_message(
                    "\t200 day moving average data required for golden cross \
                    data: Collecting 200 day  moving average data")

                DataTypes.moving_average_200(data)

            # TODO Find when they cross, mark distance beteween them
            data.golden_cross = []
            for d in data.dates:
                data.golden_cross += [data.moving_average_50.get_value(d, "MA_50") -
                                      data.moving_average_200.get_value(d, "MA_200")]
            data.golden_cross = \
                pd.DataFrame({"Golden_Cross": data.golden_cross}).set_index(data.dates)
        pass

    def pullbacks(data, historical=True):
        verbose_message("Collecting pullback Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def on_balance_volume(data, historical=True):
        verbose_message("Collecting obv Data for " + data.symbol)
        if not historical:
            pass
        else:
            data.on_balance_volume = []
            last_date = None
            for d in data.dates:
                if last_date is None:
                    data.on_balance_volume += [data.market['Volume'][d]]
                else:
                    if data.market['Close'][d] > data.market['Close'][last_date]:
                        data.on_balance_volume += [data.on_balance_volume[-1] + data.market['Volume'][d]]
                    elif data.market['Close'][d] < data.market['Close'][last_date]:
                        data.on_balance_volume += [data.on_balance_volume[-1] - data.market['Volume'][d]]
                    else:
                        data.on_balance_volume += [data.on_balance_volume[-1]]

                last_date = d
            data.on_balance_volume = \
                pd.DataFrame({"OBV": data.on_balance_volume}).set_index(data.dates)

    def pivot_point(data, historical=True):
        verbose_message("Collecting pivot point Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass
        pass

    def a_d_line(data, historical=True):
        verbose_message("Collecting a/d line Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def average_directional_index(data, historical=True):
        verbose_message("Collecting adx Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def aroon(data, historical=True):
        verbose_message("Collecting aroon Data for " + data.symbol)
        if not historical:
            pass
        else:
            if data.year_historical is None:
                verbose_message(
                    "\tYear historical data required for 26 day trailing moving \
                    average data: Collecting year historical data")

            data.aroon = []
            for d in data.dates:
                period = data.year_historical[d][LENGTH_YEAR - AROON_PERIOD:]
                period = period[::-1]
                aroon_up = 100 * (period.index(max(period))) / AROON_PERIOD
                aroon_down = 100 * (period.index(min(period))) / AROON_PERIOD
                aroon = aroon_up-aroon_down
                data.aroon += [aroon]
            data.aroon = \
                pd.DataFrame({"AROON": data.aroon}).set_index(data.dates)

    def stochastic_oscillator(data, historical=True):
        verbose_message("Collecting stochastic oscillator Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def layoff_analysis(data, historical=True):
        verbose_message("Collecting layoff analysis Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def auto_encoded_stock_info(data, historical=True):
        verbose_message("Collecting auto-encoded Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def stock_twits_feed(data, historical=True):
        verbose_message("Collecting stock twit Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def stock_twits_sentiment(data, historical=True):
        verbose_message("Collecting stock twit sentiment analysis Data for " + data.symbol)
        if not historical:
            pass
        else:
            pass

        pass

    def month_futures(data, historical=True):
        verbose_message("Collecting Month Future Data for " + data.symbol)
        if not historical:
           pass
        else:
            data.month_futures = {}
            i = 0
            closes = StockDataCollection.get_stock_close_list(data.symbol, data.str_dates[0], data.str_dates[-1])
            for d in data.dates:
                data.month_futures[d] = closes[i + 1:i + LENGTH_MONTH + 1]
                i += 1

    def week_futures(data, historical=True):
        verbose_message("Collecting Week Future Data for " + data.symbol)
        if not historical:
            pass
        else:
            if data.month_futures is None:
                verbose_message(
                    "\tMonth Futures data required for Week Futures data: Collecting Month Futures data")
                DataTypes.month_futures(data)
            data.week_futures = {}
            for d in data.dates:
                data.week_futures[d] = data.month_futures[d][:LENGTH_WEEK]

    def day_futures(data, historical=True):
        verbose_message("Collecting Day Future Data for " + data.symbol)
        if not historical:
            pass
        else:
            if data.month_futures is None:
                verbose_message(
                    "\tMonth Futures data required for Day Futures data: Collecting Month Futures data")
                DataTypes.month_futures(data)
            data.day_futures = {}
            for d in data.dates:
                data.day_futures[d] = data.month_futures[d][:LENGTH_DAY]

    def month_growth(data, historical=True):
        verbose_message("Collecting Month Growth Data for " + data.symbol)
        if not historical:
            pass
        else:
            if data.month_futures is None:
                verbose_message(
                    "\tMonth Futures data required for Month Growth data: Collecting Month Futures data")
                DataTypes.month_futures(data)
            data.month_growth = []
            for d in data.dates:
                if len(data.month_futures[d]) < LENGTH_MONTH:
                    data.month_growth += [np.nan]
                else:
                    # TODO update with proprietary algorithm determining the detremet an effect of growth
                    data.month_growth += [np.mean(data.month_futures[d]) - data.market.get_value(d,'Close')]
            data.month_growth = pd.DataFrame({"Month_Growth": data.month_growth}).set_index(data.dates)

    def week_growth(data, historical=True):
        verbose_message("Collecting Week Growth Data for " + data.symbol)
        if not historical:
            pass
        else:
            if data.week_futures is None:
                verbose_message(
                    "\tWeek Futures data required for Week Growth data: Collecting Week Futures data")
                DataTypes.week_futures(data)
            data.week_growth = []
            for d in data.dates:
                if len(data.week_futures[d]) < LENGTH_WEEK:
                    data.week_growth += [np.nan]
                else:
                    # TODO update with proprietary algorithm determining the detremet an effect of growth
                    data.week_growth += [np.mean(data.week_futures[d]) - data.market.get_value(d,'Close')]
            data.week_growth = pd.DataFrame({"Week_Growth": data.week_growth}).set_index(data.dates)

    def day_growth(data, historical=True):
        verbose_message("Collecting Day Growth Data for " + data.symbol)
        if not historical:
            pass
        else:
            if data.day_futures is None:
                verbose_message(
                    "\tDay Futures data required for Day Growth data: Collecting Day Futures data")
                DataTypes.day_futures(data)
            data.day_growth = []
            for d in data.dates:
                if len(data.day_futures[d]) < LENGTH_DAY:
                    data.day_growth += [np.nan]
                else:
                    # TODO update with proprietary algorithm determining the detremet an effect of growth
                    data.day_growth += [np.mean(data.day_futures[d]) - data.market.get_value(d,'Close')]
            data.day_growth = pd.DataFrame({"Day_Growth": data.day_growth}).set_index(data.dates)

    # TODO add methods to get short mid and longterm gains
    # TODO add methods to get past Gains
    # TODO add algorithm to calcualate any type of gains and associated risk in terms of volitility of gains and such

    # methods that are collected when all are requested
    ALL = [google_trends, ad_meter, year_data, month_data, week_data, twitter_feed, twitter_sentiment, reddit_feed,
           reddit_sentiment, press_release_feed, press_release_sentiment, moving_average_200, moving_average_50,
           trailing_moving_average_12, trailing_moving_average_26, volatility, relative_strength_index,
           moving_average_convergence_divergence, golden_cross, pullbacks, on_balance_volume, pivot_point, a_d_line,
           average_directional_index, aroon, stochastic_oscillator, layoff_analysis, auto_encoded_stock_info,
           stock_twits_feed, stock_twits_sentiment,month_futures, week_futures, day_futures, month_growth, week_growth,
           day_growth]


# TODO maybe make all dates used and use krogh interpolation to compnsate Still hold a list of market open days tho
class Universe(object):

    def add_stock(self, symbol):
        """
        adds a stock to the unverse and collects data
        just recollects data if stock is already in the universe
        :param symbol: the symbol of the stock to add
        :return: None
        """
        verbose_message("Adding " + symbol + "...")
        if symbol not in self.stocks:
            self.stocks += [symbol]

        data = StockData()

        data.name = StockDataCollection.get_stock_name(symbol)
        data.symbol = symbol
        data.market = StockDataCollection.get_market_data(symbol,
                                                          str(self.start_date)[:USEFUL_TIMESTAMP_CHARS],
                                                          str(self.end_date)[:USEFUL_TIMESTAMP_CHARS])

        # create a list of dates in the YYYY-MM-DD format
        data.str_dates = [str(i)[:USEFUL_TIMESTAMP_CHARS] for i in list(data.market.index)]
        data.dates = data.market.index

        for i in data.dates:
            if i not in self.dates:
                self.dates += [i]
                self.dates.sort()
                self.str_dates = [str(i)[:USEFUL_TIMESTAMP_CHARS] for i in list(self.dates)]

        for collection_function in self.features:
            collection_function(data)

        data.position = []

        for _ in data.dates:
            data.position += [0]
            self.cash += [self.starting_capital]

        data.position = pd.DataFrame({"Position": data.position}).set_index(data.dates)

        self.cash = pd.DataFrame({"cash": self.cash}).set_index(data.dates)
        debug_message(data)
        self.stock_data[symbol] = data

    def get_back_dataframe(self, end_date=None, stocks=None):
        """
        gets the data up to the date
        :param end_date: date to get data up to
        :param stocks: list of stocks to get the back data for None gets all
        :return: StockData
        """
        if end_date is None:
            end_date = self.dates[-1]

        if type(end_date) is not datetime.datetime and type(end_date) is not pd.tslib.Timestamp:
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        if stocks is None:
            stocks = self.stocks

        info = {}
        for stock in stocks:
            info[stock] = self.stock_data[stock].to_stock_dataframe_range(start_date=None, end_date=end_date)

        return info

    def get_back_data(self, end_date=None, stocks=None):
        """
        gets the data up to the date
        :param end_date: date to get data up to
        :param stocks: list of stocks to get the back data for None gets all
        :return: StockData
        """
        if end_date is None:
            end_date = self.dates[-1]

        if type(end_date) is not datetime.datetime and type(end_date) is not pd.tslib.Timestamp:
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        if stocks is None:
            stocks = self.stocks

        info = {}
        for stock in stocks:
            info[stock] = self.stock_data[stock].to_stock_data_range(start_date=None, end_date=end_date)

        return info

    def collect_all_stock_data(self):
        """
        collects data for all stocks in universe
        :return: None
        """
        for stock in self.stocks:
            self.add_stock(stock)

    def __init__(self, stocks=None, start_date='FiveYear', end_date='Today', features=None, verbose=False, capital=0):
        """
        initializes the stock universe
        :param stocks: list of stocks to add
        :param start_date: the date to start collection on 'FiveYear' for 5 years of data
        :param end_date: the date to end collection on 'Today' for today's date
        :param features: list of features to collect data for.
        """

        # Set default features
        if type(features) is not list:
            features = [features]

        if features is None:
            features = []

        if DataTypes.ALL in features:
            features = DataTypes.ALL

        # set variables for a stock universe
        self.verbose = verbose
        self.stocks = stocks
        self.features = features
        self.stock_data = {}
        end_date = datetime.datetime.today() if end_date == 'Today' \
            else datetime.datetime.strptime(end_date, "%Y-%m-%d")

        if type(end_date) is not datetime.datetime and type(end_date) is not pd.tslib.Timestamp:
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        self.end_date = end_date

        start_date = end_date - datetime.timedelta(365 * 5 + 1) if start_date == 'FiveYear' \
            else datetime.datetime.strptime(start_date, "%Y-%m-%d")

        if type(start_date) is not datetime.datetime and type(start_date) is not pd.tslib.Timestamp:
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

        self.start_date = start_date

        self.date = start_date  # initial date that the stock universe is on

        # create a list of dates in the YYYY-MM-DD format
        self.str_dates = []
        self.dates = []

        self.starting_capital = capital
        self.cash = []

        self.collect_all_stock_data()
        self.unique_data = {}
        self.shuffled_data_reset()
        # TODO add ability to order stocks and build a profile having total percent returns as well as capital
        # TODO have ability to select types of data to get fundementals, trends, stock twits anal,
        # TODO ad meter, past prices and volumes, twitter reddit and press releases

    def get_data(self, stock=None):
        """
        gets data for all stocks or a specific stock
        :param stock: the stock to collect data for None gets data for all
        :return: the data for the stock(s)
        """
        if stock is None:
            return self.stock_data
        else:
            return self.stock_data[stock]

    def get_data_date(self, date):
        """
        gets the data on a given date
        :param date: the date to get data for
        :return: StockData for the day
        """
        data = {}
        for stock in self.stocks:
            data[stock] = self.stock_data[stock].to_stock_dataframe_day(date)
        return data

    def collect_data_date(self, date=None):
        """
        recollects data for a given date
        :param date: date to collect data for. Universe today by default
        :return: None
        """
        if date is None:
            date = self.date
        # TODO make it so it doenst re-collect all data and just adds historical's data
        self.collect_all_stock_data()

    def get_date(self):
        """
        gets the current date
        :return: the current date of the universe
        """
        return self.date

    def plot_data(self, symbols, columns, start_date=None, end_date=None):
        """
        uses matplotlib to plot the data collected
        :param symbols: the symbols to plot
        :param columns: list of columns to plot
        :param start_date: starting date to plot None for beginning
        :param end_date: ending date to plot None for end
        :return: None
        """
        # TODO plot stock data for given symbols, maybe on a single graph
        pass

    def get_data_dates(self, symbol, start_date, end_date):
        pass

    def update_data(self, stocks=None):
        """
        updates all data for stocks
        :param stocks: updates for all if none, or for a list of stocks
        :return: None
        """
        if stocks is None:
            self.collect_all_stock_data()
            return
        for stock in stocks:
            self.add_stock(stock)

    def remove_stock(self, stock):
        """
        removes a stock from the universe
        :param stock: symbol of the stock to remove
        :return: None
        """
        if stock in self.stocks:
            self.stocks.remove(stock)
        if stock in self.stock_data.keys():
            del self.stock_data[stock]

    def order_stock(self, stock, amount, date=None):
        """
        orders an amount of stocks if is able to be done
        :param stock: the name of stock to order
        :param amount: amount of the stock to buy
        :param date: date to buy on
        :return: true if can be bought false if it is not purchased
        """
        # TODO add functionality to fail a stock purchase if there is not enough funds available to buy
        if date is None:
            date = self.date

        if type(date) is not datetime.datetime and type(date) is not pd.tslib.Timestamp:
            date = datetime.datetime.strptime(date, "%Y-%m-%d")

        cost = self.stock_data[stock].market['Close'][date] * (amount -
                                                               self.stock_data[stock].position['Position'][date])

        self.stock_data[stock].position.set_value(date, 'Position', amount)

        self.cash.set_value(date, "cash", self.cash.get_value(date, 'cash') - cost)
        last_date = None
        for d in self.dates[self.dates.index(date):]:
            if last_date is None:
                last_date = d
            else:
                self.stock_data[stock].position['Position'][d] = self.stock_data[stock].position['Position'][last_date]
                self.cash['cash'][d] = self.cash['cash'][last_date]
        return True

    def buy_stock(self, stock, amount, date=None):
        """
        buys an amount of stocks if is able to be done
        :param stock: the name of stock to order
        :param amount: amount of the stock to buy
        :param date: date to buy on
        :return: true if can be bought false if it is not purchased
        """
        if date is None:
            date = self.date

        if type(date) is not datetime.datetime and type(date) is not pd.tslib.Timestamp:
            date = datetime.datetime.strptime(date, "%Y-%m-%d")

        self.order_stock(stock, self.stock_data[stock].position['Position'][date] + amount, date)

    def sell_stock(self, stock, amount, date=None):
        """
        sells an amount of stocks if is able to be done
        :param stock: the name of stock to order
        :param amount: amount of the stock to buy
        :param date: date to buy on
        :return: true if can be bought false if it is not purchased
        """
        if date is None:
            date = self.date

        if type(date) is not datetime.datetime and type(date) is not pd.tslib.Timestamp:
            date = datetime.datetime.strptime(date, "%Y-%m-%d")

        self.order_stock(stock, self.stock_data[stock].position['Position'][date] - amount, date)

    # TODO some randomness to buys so they do not always happen at high and low points

    def get_day_returns(self, stocks=None, date=None):
        """
        gets value for the day based on stock holdings
        :param stocks: stocks to get evaluation on
        :param date: date to get the evaluated price for None for today in universe
        :return: money in stocks
        """
        if stocks is None:
            stocks = self.stocks

        if date is None:
            date = self.date

        if type(date) is not datetime.datetime and type(date) is not pd.tslib.Timestamp:
            date = datetime.datetime.strptime(date, "%Y-%m-%d")

        stock_money = 0
        for stock in stocks:
            stock_day = self.stock_data[stock]
            # TODO find a better way than avging open and cloase
            stock_money += stock_day.position['Position'][date] *\
                           (stock_day.market['Close'][date] + stock_day.market['Open'][date])/2

        return stock_money

    def run_back_test(self, algorithm):
        """
        runs the back test algorithm to
        :param algorithm: algorithm that takes [backData(data up to the day), original order (type Order)] into account
        :return: total returns for each day in the back test
        """
        for date in self.dates:
            info = dict()
            info['cash'] = self.cash['cash'][date]
            info['date'] = date
            original_order = Order()
            for stock in self.stocks:
                original_order.order(stock, self.stock_data[stock].position['Position'][date])
            make_order = algorithm(self.get_back_dataframe(date), original_order, info)

            for stock in make_order.orders:
                self.order_stock(stock, make_order.orders[stock], date)

    def get_returns(self, start_date=None, end_date=None, stocks=None):
        """
        gets the returns over a peroid of time for a set of stocks
        :param start_date: the strat date to get profits
        :param end_date: end date to calcualte profits
        :param stocks: the list of stocks to get the profits for
        :return: DataFrame of profits for every day
        """
        if stocks is None:
            stocks = self.stocks

        if start_date is None:
            start_date = self.dates[0]

        if end_date is None:
            end_date = self.dates[-1]

        if type(end_date) is not datetime.datetime and type(end_date) is not pd.tslib.Timestamp:
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        if type(start_date) is not datetime.datetime and type(start_date) is not pd.tslib.Timestamp:
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

        dates_to_check = self.dates[self.dates.index(start_date): self.dates.index(end_date) + 1]

        stock_money = []

        for date in dates_to_check:
            stock_money += [self.get_day_returns(stocks, date)]

        stock_money = pd.DataFrame({"stock value": stock_money}).set_index([self.dates])

        return_info = join_features(stock_money, self.cash)
        return_info['value'] = return_info['cash'] + return_info['stock value']

        return return_info

    def set_capital(self, amount):
        """
        sets the amount of capatal owned
        :param amount: amount of capital to set
        :return: None
        """
        self.starting_capital = amount

    def add_feature(self, feature):
        """
        adds a feature for data to collect
        :param feature: DataTypes method to collect a feature
        :return: None
        """
        self.features += [feature]
        for stock in self.stocks:
            feature(self.stock_data[stock])

    def shuffled_data_avalible(self):
        """
        gets the remaining number of data points available to retrieve
        :return: number of data points left
        """
        available = 0

        for stock in self.unique_data.keys():
            available += len(self.unique_data[stock])

        return available

    def shuffled_data_reset(self):
        """
        resets the shuffled data to be full of all dates and stocks
        :return: None
        """
        self.unique_data = {}
        for stock in self.stocks:
            self.unique_data[stock] = []
            for date in self.dates:
                self.unique_data[stock] += [date]

    def shuffled_data_get(self):
        """
        gets unique data from any stock at any time period, each is only used
        once before it is dropped from the data getter.
        :return: data slice
        """

        if self.unique_data == {}:
            self.shuffled_data_reset()
        keys = list(self.unique_data.keys())[:]
        random.shuffle(keys)
        stock = keys[0]
        dates = list(self.unique_data[stock])[:]
        random.shuffle(dates)
        date = dates[0]
        self.unique_data[stock].remove(date)
        data = self.get_back_data(date, [stock])

        # Cleanup
        delete = None
        for stock in self.unique_data.keys():
            if len(self.unique_data[stock]) == 0:
                delete = stock
        if delete is not None:
            del self.unique_data[delete]

        return data

    def __next__(self):
        """
        Moves to the next day
        :return: true if next day is found false if None
        """
        found = False
        for i in self.dates:
            if self.date == i:
                found = True
            if found:
                for stock in self.stocks:
                    # update positions to cary over to next day
                    self.stock_data[stock].position['Position'][i] =\
                        self.stock_data[stock].position['Position'][self.date]

                    self.cash['cash'][i] = self.cash['cash'][self.date]

                self.date = i
                return True
        return False

    def __getitem__(self, item):
        """
        gets a stocks data
        :param item: stock symbol
        :return: the data for that stock
        """
        return self.get_data(stock=item)

    def __delitem__(self, key):
        self.remove_stock(stock=key)

    def __len__(self):
        return len(self.stocks)

    def __contains__(self, item):
        return item in self.stock_data.keys()


if __name__ == '__main__':
    a = Universe(['MMM'], "2012-03-05", "2012-06-05", features=DataTypes.ALL, capital=10000)
    # a.order_stock('MMM', 10, "2012-03-14")

    def alg(data, order, info):
        if data['MMM']['Day_Growth'][info['date']] > 0:
            order.buy('MMM', int(info['cash']/data['MMM']['Close'][info['date']]))
        else:
            order.sell('MMM', data['MMM']['Position'][info['date']])

        return order
    while a.shuffled_data_avalible() > 0:
        print("Available:", a.shuffled_data_avalible())
        print(a.shuffled_data_get()['MMM'])

    # a.run_back_test(alg)
    # rets = a.get_returns()
    #
    # print(rets)
    # rets.plot()
    # plt.show()
