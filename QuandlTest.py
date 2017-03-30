import quandl
quandl.ApiConfig.api_key = "PqbW_ePwH8ijgmhysKAZ"
mydata = quandl.get_table('WIKI/PRICES', ticker='AAPL')


print(mydata)