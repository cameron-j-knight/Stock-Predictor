import pytrends.request as request
import base64
import  matplotlib.pyplot as plt
from yahoo_finance import Share
import datetime
import pandas as pd
import time

userName = "camknightt@gmail.com"
password = base64.b16decode("4C316C6169746569676F21676F")
password = str(password)[2:-1]


def CollectStockData(stock):
    today = datetime.date.today()
    five_years = today - datetime.timedelta(365*5 + 1)
    for i in range(3):

        try:
            share = Share(stock)
            if share.get_name() != None:
                longStock = share.get_historical(str(five_years), str(today))[::-1]
                longStock = pd.DataFrame(longStock)
                longStock = longStock.set_index(pd.DatetimeIndex(longStock['Date']))

                del longStock['Date']
                del longStock['Symbol']
                longStock = longStock.apply(pd.to_numeric, errors='coerce')
                return longStock , share.get_name()
        except:
            print("Data not collected. Trying Again")
    return None,share.get_name()


def CollectStockTrends(name,related=True):
    name = name.split(',')[0]
    Req = request.TrendReq(userName,password,"PyTrends")
    for i in range(len(name.split(' ')),0,-1):
        try:
            if (len(Req.suggestions(name)) > 0):
                tryName = Req.suggestions(sugName)[0]['title']
                goodName = 0
                for w in sugName.split(' '):
                    if w in tryName:
                        goodName += 1
                if(goodName > (sugName.split(' ')/ 3) ):
                    sugName = tryName
                    print(tryName)

            Req.build_payload([sugName])
            Req.interest_over_time()
        except:
            sugName = (''.join([x + ' ' for x in name.split(' ')[:i]])).strip()
            print(sugName)
        else:
            break
    print(sugName)
    Req.build_payload([sugName])
    #TODO Try and get better results using this techniques
    try:
        related = Req.related_queries()
        related = list(related[sugName]['top']['query'])
        new_related = []
        for i in related:
            try:
                sugs = Req.suggestions(i)
                if(len(sugs) > 0):
                    new_related += [sugs[0]['title']]
            except:
                pass
        related = list(set(new_related))
        relWords = []
        for x in related:
            try:
                Req.build_payload([x])
                relWords += [Req.interest_over_time()]
            except:
                print('Failed', x)
        data = relWords[0]
        relWords = relWords[1:]
        for i in relWords:
            data = data.join(i,rsuffix='_n',how='outer')

        return data

    except:
        print('unable to find extra data on Stock')

    try:
        return Req.interest_over_time()
    except:
        print("Failed to retrieve any data from company")
        return pd.DataFrame()


def CollectStockFeatures(symbol):

    data,name = CollectStockData(symbol)
    print(name)
    trendData = CollectStockTrends(name,True)

    #manipulate the dataframes to fill in NANs
    dates = pd.DataFrame(data['Close'])
    trendData = dates.join(trendData,rsuffix='_n',how='outer')
    trendData = trendData.fillna(method='ffill')  #fill trend data with old values
    del trendData['Close']

    data = data.join(trendData,rsuffix='_n',how='outer')

    data = data.dropna()

    return data



data = CollectStockFeatures("X")

relevantData = pd.DataFrame(data['Close'])
relavantTrends = pd.DataFrame(data.iloc[:,6:])

print(data.corr())
relevantData = relevantData.join(relavantTrends)

a = relevantData.plot()
plt.show()

#print(data)

# df.plot();
# plt.show()



# for i in range(100):
#     with open('TradableStocks.txt','a+') as nf:
#         with open('Stocks.txt') as f:
#             for line in f:
#                 print('='*255)
#                 symb = line.split('|')[0]
#                 data,name = CollectStockData(symb)
#                 print(name)
#                 try:
#                     CollectStockTrends(name,True)
#                 except:
#                     print('Failed')
#                 else:
#                     print('Sucess')
#                     if(data is not None):
#                         print("Recorded: ", symb)
#                         nf.write(symb+'\n')
#                         nf.flush()
#










