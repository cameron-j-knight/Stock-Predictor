import Universe as uv
import pandas as pd
import pickle

import tensorflow as tf

start_date = "2005-01-01"
end_date = "2017-01-01"

company_list = "companylist.csv"
growth_save = "data/GrowthData.pkl"
companies = pd.read_csv(company_list)['Symbol'].tolist()

try:
    data = pickle.load(open(growth_save, "rb"))
except FileNotFoundError:
    features = [uv.DataTypes.month_growth, uv.DataTypes.week_growth, uv.DataTypes.day_growth]
    data = uv.Universe([], start_date, end_date, features=features)

    pickle.dump(data, open(growth_save, 'wb'))
finally:
    end_collect = -1
    for i in range(len(companies)):
        if companies[i] in data:
            end_collect = i
    i = 0
    data_modified = False
    for symbol in companies[end_collect+1:]:
        try:
            print("collecting: " + symbol)
            data.add_stock(symbol)
            data_Modified = True
        except:
            print("Failed: " + symbol)

        if i % 25 == 0 and data_modified:
            print("Saving..........")
            pickle.dump(data, open(growth_save, 'wb'))
        i += 1

    if data_modified:
        print("Saving..........")
        pickle.dump(data, open(growth_save, 'wb'))

# label as sell buy or keep

# [day growth, month, growth, week growth, stocks held]
x = tf.placeholder(tf.float32, [None, 4])
