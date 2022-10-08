import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import mplfinance as mpf
import datetime
from sklearn.metrics import silhouette_score

# ----------------- inputs ---------------------

# --- calculate based on selected timeframe
# Valid periods: [1mo,1wk,5m,3mo,6mo,1y,5y,Max]
period = "Max"
# Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
interval = "1wk"

# ticker
symbol = 'BNB-USD'
# select range
rangee = 1   # on off
fromm = 184
too = 666

# number of clusters +1
numK = 7

# K factor 
# more about K-value - https://datascience.stackexchange.com/questions/75789/why-is-10-considered-the-default-value-for-k-fold-cross-validation
# 10 is considered as default value 
# Basically if you have enough data, the factor can be lowered - https://datascience.stackexchange.com/questions/75789/why-is-10-considered-the-default-value-for-k-fold-cross-validation
n_init_input = 4

# -----------------------------------------------

def load_prices(ticker, per, interval):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    if rangee == 1:
        data.drop(data[data['Low'] <= fromm].index, inplace = True)
        data.drop(data[data['High'] <= fromm].index, inplace = True)
        data.drop(data[data['Close'] <= fromm].index, inplace = True)
        data.drop(data[data['Open'] <= fromm].index, inplace = True)
        
        data.drop(data[data['Low'] >= too].index, inplace = True)
        data.drop(data[data['High'] >= too].index, inplace = True)
        data.drop(data[data['Close'] >= too].index, inplace = True)
        data.drop(data[data['Open'] >= too].index, inplace = True)
    # remove zero values from  df
    data.drop(data[data['Low'] == 0].index, inplace = True)

    #print(data)
    return data.dropna()

def get_optimum_clusters(df, saturation_point=0.05):
    '''
    :param df: dataframe
    :param saturation_point: The amount of difference we are willing to detect
    :return: clusters with optimum K centers
    
    This method uses elbow method to find the optimum number of K clusters
    We initialize different K-means with 1..10 centers and compare the inertias
    If the difference is no more than saturation_point, we choose that as K and move on
    '''
    wcss = []
    k_models = []
    size = min(numK, len(df.index))
    for i in range(1, size):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=n_init_input, random_state=None)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        k_models.append(kmeans)

    # View inertia - good for electing the saturation point
    print(wcss)

    # Compare differences in inertias until it's no more than saturation_point
    optimum_k = len(wcss)-1
    for i in range(0, len(wcss)-1):
        diff = abs(wcss[i+1] - wcss[i])
        if diff < saturation_point:
            optimum_k = i
            break

    print("Optimum K is " + str(optimum_k + 1))
    optimum_clusters = k_models[optimum_k]

    return optimum_clusters
    

data = load_prices(symbol, period, interval)

lows = pd.DataFrame(data=data, index=data.index, columns=["Low"])
highs = pd.DataFrame(data=data, index=data.index, columns=["High"])

low_clusters = get_optimum_clusters(lows)
low_centers = low_clusters.cluster_centers_
low_centers = np.sort(low_centers, axis=0)

high_clusters = get_optimum_clusters(highs)
high_centers = high_clusters.cluster_centers_
high_centers = np.sort(high_centers, axis=0)

# How good are the clusters?
low_score=silhouette_score(lows,low_clusters.labels_)
high_score=silhouette_score(highs,high_clusters.labels_)
print(f"Silhouette score Lows: {low_score} Highs: {high_score}")


lowss = []
highss = []
finals = []

rounding_factor = 2

for i in low_centers:
  i = round(float(i),rounding_factor)
  lowss.append(i)

for i in high_centers:
  i = round(float(i),rounding_factor)
  highss.append(i)

print('lows/support: ', lowss)
print('highs/resistance: ', highss)
a = len(lowss)
b = len(highss)

y = 0

print(f'indicator("My script", overlay=true)')

x = datetime.datetime.now()
nametitlet = x.strftime("%d-%m-%Y")

file_object = open(symbol+str(nametitlet)+str(interval)+str(period)+str(n_init_input)+".txt", 'a+')
zline = '//@version=5'
fline = 'indicator("SR", overlay=true)'
sline = "plot(close)"
file_object.write(zline+"\n")
file_object.write(fline+"\n")
file_object.write(sline+"\n")
for i in range(a):
    a = 'hline('+str(lowss[y])+', title='+'"'+interval+'_'+symbol+'_support'+'"'', color=color.rgb(56, 146, 71), linestyle=hline.style_solid, linewidth=2)'
    # insert supports into the file
    file_object.write(a+"\n")
    print(a)
    #print(side)
    y += 1


z = 0
for i in range(b):
    b = 'hline('+str(highss[z])+', title='+'"'+interval+'_'+symbol+'_resistance'+'"'', color=color.rgb(192, 48, 48), linestyle=hline.style_solid, linewidth=2)'
    # insert resistances into the file
    file_object.write(b+"\n")
    print(b)
    #print(side)
    z += 1
file_object.close()

# Plotting
plt.style.use('fast')
ohlc = data.loc[:, ['Open', 'High', 'Low', 'Close']]
fig, ax = mpf.plot(ohlc.dropna(), type = 'candle', style='charles', show_nontrading = False,returnfig = True,
                  ylabel='Price',title=symbol)

for low in low_centers[:9]:
    ax[0].axhline(low[0], color='green', ls='-', alpha=.2)

for high in high_centers[-9:]:
    ax[0].axhline(high[0], color='red', ls='-', alpha=.1)

plt.show()
