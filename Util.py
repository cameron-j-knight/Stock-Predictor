"""
Author: Cameron Knight
"""
import pandas as pd

# Constants
FETCH_ATTEMPTS = 3              # number of attempts until failure is sent
LENGTH_DAY = 1                  # number of days in one day
LENGTH_YEAR = 365 * LENGTH_DAY  # number of days to consider a year
LENGTH_MONTH = 30 * LENGTH_DAY  # number of days to consider a month
LENGTH_WEEK = 5 * LENGTH_DAY    # number of days to consider a week
USEFUL_TIMESTAMP_CHARS = 10     # number of characters that are useful in a timestamp
FAIL_DELAY = 60                 # amount of time taken after a failure in case of too many calls
DEBUG = False
VERBOSE = False
RSI_PERIOD = 14                 # number of days to use as the period for rsi calculation
AROON_PERIOD = 25                # number of days to base aroon oscillator calculation on.
# Username Passwords Keys
trends_username1 = "camknightt@gmail.com"
trends_password1 = "L1laiteigo!go"

trends_username2 = "jeffeled@gmail.com"
trends_password2 = "Avpr1423"

stock_list = "Stocks.txt"

# Utility Methods
class FillMethod:
    """
    Methods that can be used to fill nans
    """
    ZEROS = 'zeros'                            # fill nans with 0
    ONES = 'ones'                              # fill nans with 1
    INTERPOLATE = 'interpolate'                # fill nans with interpolated old and new
    FORWARD_FILL = 'ffill'                     # fill nans with last available value then do backward fill
    BACKWARD_FILL = 'bfill'                    # fill nans with next available value then do forward fill
    MEAN = 'mean'                              # fill nans with mean value
    REMOVE = 'remove'                          # deletes all nans
    KROGH = 'krogh'                            # use past data to predict trends
    FUTURE_KROGH = 'future krogh'              # uses krogh's algorithm to only predict forward
    NONE = 'none'                              # do nothing with the nans


def future_krogh(df):
    """
    uses krogh's algorithum to predict nans using only past data
    :param df: df to predict
    :return: new dataframe that has been predicted
    """
    # TODO make it so the algorithm fills in first blank then re reuns and fills second blank and so on
    # TODO make it take past data that can be used to get futures with more accuraccy building o that initial
    new_df = pd.DataFrame()
    for column in df:
        print(column)
        indexes = [0]
        for i in range(1, len(df[column])):
            print(df[column][i])
            if pd.isnull(df[column][i-1]) and not pd.isnull(df[column][i]):
                indexes += [i]
        indexes += [-1]
        dfs = []
        for i in range(1, len(indexes)):
            dfs += [df[column][indexes[i-1]:indexes[i]]]
        for i in range(len(dfs)):
            dfs[i] = dfs[i].interpolate('krogh')
        new_df[column] = pd.concat(dfs)
    return new_df


def join_features(df_a, df_b, fill_method=FillMethod.ZEROS, dates=None):
    """
    utility function to join two pandas dataframes while filling in un obtained data with
    copies of old data to fill in the blanks
    :param df_a: the data frame to add to
    :param df_b: the data frame to add from
    :param fill_method the method in which to fill the nans default fill with zeros
    :param dates: list of dates that should be kept
    :return: a data frame of combined df_a and df_b with dates all NA will be filled with linear
    interpolation between old and new data
    """

    data = df_a.join(df_b, rsuffix='_n', how='outer')
    if fill_method == FillMethod.ZEROS:
        data = data.fillna(0)

    elif fill_method == FillMethod.ONES:
        data = data.fillna(1)

    elif fill_method == FillMethod.INTERPOLATE:
        data = data.interpolate()
        data = data.fillna(method='ffill')
        data = data.fillna(method='bfill')

    elif fill_method == FillMethod.FORWARD_FILL:
        data = data.fillna(method='ffill')
        data = data.fillna(method='bfill')

    elif fill_method == FillMethod.BACKWARD_FILL:
        data = data.fillna(method='bfill')
        data = data.fillna(method='ffill')

    elif fill_method == FillMethod.MEAN:
        data = data.fillna(data.mean())

    elif fill_method == FillMethod.KROGH:
        data = data.interpolate('krogh')

    elif fill_method == FillMethod.FUTURE_KROGH:
        data = future_krogh(data)

    elif fill_method == FillMethod.REMOVE:
        data = data.dropna(axis=0, how='any', inplace=True)

    if dates is not None:

        thees_dates = data.index.values.tolist()

        remove_dates = [i for i in thees_dates if i in dates]

        data.drop(remove_dates)

    return data


def verbose_message(message):
    """
    prints out an informative message if program is run in verbose mode
    :param message: the message to print
    :return: None
    """
    if VERBOSE:
        print(message)


def debug_message(message):
    """
    prints out a debug method if program is in debug mode
    :param message: the message to print
    :return: None
    """
    if DEBUG:
        print(message)
