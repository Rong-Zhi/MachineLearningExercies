import pandas as pd
import numpy as np
import quandl, math
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")

# Using the difference of data to reduce number of features(grab useful data)
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PTC_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PTC_change', 'Adj. Volume']]

# print(df.head())

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True) # filling the missed data with -99999
forecast_out = int(math.ceil(0.01 * len(df)))
# print(df)

df['label'] = df[forecast_col].shift(-forecast_out) # shift the forecast to 16 days ago

X = np.array(df.drop(['label'], 1)) # features
y = np.array(df['label'])   # label

X = preprocessing.scale(X) # transfer data into -1 to 1
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True) # drop any still NaN info

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# # SVM classifier
# clf = svm.SVR()
# clf.fit(X_train,y_train)    # fitting the training data
# confidence = clf.score(X_test, y_test) # test
# # print(confidence)

# Linear Regression classifier
clf = LinearRegression()
# clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
# print(confidence)

forecast_set = clf.predict(X_lately)
# print(forecast_set, confidence, forecast_out)

# # kernel
# # rbf- (Gaussian) radial basis function kernel
# for k in ['linear', 'poly', 'rbf', 'sigmoid']:
#     clf = svm.SVR(kernel=k)
#     clf.fit(X_train,y_train)
#     confidence = clf.score(X_test, y_test)
#     print(k,confidence)

style.use('ggplot')

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()