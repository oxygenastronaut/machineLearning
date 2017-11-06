import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.01 * len(df)))  # this predicts out 1% of the dataframe. 
# it gets whatevers days price using data from ten days ago
# this is form the 0.1 
# days in advance
df['label']  = df[forecast_col].shift(-forecast_out)	# this shifts the columns negatively
# this label col for each row will be the adj close price ten days into the future
# so the features are what may cause the adj close price in 1% days to change
# you wont get rich off of this current model but its not too bad

X = np.array(df.drop(['label'], 1)) 	# df.drop returns a new df without the label col as a numpy array
# now to scale X
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]		# this is what is predicted against
X = [::-forecast_out]

df.dropna(inplace = True)
y = np.array(df['label'])
y = np.array(df['label'])

# create training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)


