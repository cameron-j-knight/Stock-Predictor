import datetime
from Util import *
import StockDataCollection
import numpy as np

"""
Author: Cameron Knight
Copyright 2017, Cameron Knight, All rights reserved.
"""
class StockData(object):
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
        self.week_futures= None
        self.month_futures = None

    def all_scalar_data(self):
        vars = [i for i in dir(self) if not callable(getattr(self, i)) and
                not i.startswith("__") and
                isinstance(i, pd.DataFrame)]

    def get_day_data(self, date):
        # TODO Make this get the data for the day
        for point in self.__dict__:
            if point is not None:
                print(point, date)

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

            data.volatility = pd.DataFrame({"volatility": data.volitility}).set_index(data.dates)

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

        self.str_dates = data.str_dates
        self.dates = data.dates

        for collection_function in self.features:
            collection_function(data)

        debug_message(data)
        self.stock_data[symbol] = data

    def collect_all_stock_data(self):
        """
        collects data for all stocks in universe
        :return: None
        """
        for stock in self.stocks:
            self.add_stock(stock)

    def __init__(self, stocks, start_date='FiveYear', end_date='historical', features=None):
        """
        initializes the stock universe
        :param stocks:
        :param start_date:
        :param end_date:
        :param features:
        """

        # Set default features
        if features is not list:
            features = [features]

        if features is None:
            features = []

        if DataTypes.ALL in features:
            features = DataTypes.ALL

        # set variables for a stock universe
        self.stocks = stocks
        self.features = features
        self.stock_data = {}
        self.end_date = datetime.datetime.historical() if end_date == 'historical' \
            else datetime.datetime.strptime(end_date, "%Y-%m-%d")

        self.start_date = self.end_date - datetime.timedelta(365 * 5 + 1) if start_date == 'FiveYear' \
            else datetime.datetime.strptime(start_date, "%Y-%m-%d")

        self.date = start_date  # initial date that the stock universe is on

        # create a list of dates in the YYYY-MM-DD format
        self.str_dates = []
        self.dates = []
        self.collect_all_stock_data()

        # TODO add ability to order stocks and build a profile having total percent returns as well as capital

        # TODO have ability to select types of data to get fundementals, trends, stock twits anal,
        # TODO ad meter, past prices and volumes, twitter reddit and press releases

    def get_data(self, stock=None):
        if stock is None:
            return self.stock_data
        else:
            return self.stock_data[stock]

    def get_data_date(self, date):
        pass

    def get_data_historical(self):
        # TODO make it so it doenst re-collect all data and just adds historical's data
        self.collect_all_stock_data()

    def get_date(self):
        return self.date

    def __next__(self):
        found = False
        for i in self.dates:
            if self.date == i:
                found = True
            if found:
                self.date = i
                return

    def plot_data(self, symbols, columns):
        # TODO plot stock data for given symbols, maybe on a single graph
        pass

    def get_data_dates(self, start_date, end_date):
        pass

    def update_data(self, stock=None):
        pass

    def remove_stock(self,stock):
        if stock in self.stocks:
            self.stocks.remove(stock)
        if stock in self.stock_data.keys():
            del self.stock_data[stock]

    def __getitem__(self, item):
        return self.get_data(stock=item)

    def __delitem__(self, key):
        self.remove_stock(stock=key)

    def __len__(self):
        return len(self.stocks)

if __name__ == '__main__':
    a = Universe(['MMM'], "2012-03-05", "2012-06-05", features=DataTypes.ALL)
    a.add_stock('X')
    print(len(a))
