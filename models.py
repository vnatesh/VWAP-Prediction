import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import r2_score
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



df = pd.read_csv('/Users/vikasnatesh/Downloads/labeled_data_10s.csv')



X = df.loc[:, ~df.columns.isin(['vwap_d','vwap_v'])].values
y = df['vwap_d'].values

# Train Test split (80% train, 20% test)
X_train = X[:int(0.8*len(y))]
X_test = X[int(0.8*len(y)):]
y_train = y[:int(0.8*len(y))]
y_test = y[int(0.8*len(y)):]


def group_prob(x):
    for i in range(len(quantile)-1):
        if x>=quantile[i] and x<quantile[i+1]:
            return i+1
        if x==quantile[i+1]:
            return 10

#########   LogisticRegression
# clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

####### Random Forest

clf = RandomForestClassifier(n_estimators=20, max_depth=3,random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


pred_prob = clf.predict_proba(X_test)
prob_down = [x[0] for x in pred_prob]
prob_df = pd.DataFrame()
prob_df['label'] = y_test
prob_df['prob_down'] = prob_down

global quantile
quantile = np.percentile(prob_down,list(list(range(0,101,10))))

prob_df['group'] = prob_df['prob_down'].apply(group_prob)

pd = []
td = []
for i in range(1,11):
    group_df = prob_df[prob_df['group']==i]
    true_down_prob = stats.itemfreq(group_df['label'])[0][1]/group_df.shape[0]
    print('group:',i,'predict_prob:',group_df['prob_down'].median(),'true_prob:',true_down_prob)
    pd.append(group_df['prob_down'].median())
    td.append(true_down_prob)

print('mse',mean_squared_error(pd,td))



############ Lasso model ###############


from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv('/Users/vikasnatesh/Downloads/labeled_data_10s.csv')
# Train Test split (80% train, 20% test)

X = df.loc[:, ~df.columns.isin(['vwap_d','vwap_v'])].values

y = df['vwap_v'].values
y_train = y[:int(0.8*len(y))]
y_test = y[int(0.8*len(y)):]

X_train = X[:int(0.8*len(y))]
X_test = X[int(0.8*len(y)):]

clf = linear_model.Lasso(alpha=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('mse',mean_squared_error(y_test, y_pred))
print('R^2 score', r2_score(y_test, y_pred))

pyplot.plot(y_test, label='actual vwap')
pyplot.plot(y_pred, label='predicted vwap')
pyplot.xlabel('time (10s)')
pyplot.ylabel('normalized vwap')
pyplot.title('AAPL 10s VWAP')
pyplot.legend()
pyplot.show()




############ LSTM model ###############

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load dataset
dataset = read_csv('/Users/vikasnatesh/Downloads/labeled_data_10s.csv', header=0, index_col=1)


# values = dataset.values
# # specify columns to plot
# groups = [0, 1, 2, 3, 5, 6, 7,8,9,10]
# i = 1
# # plot each column
# pyplot.figure()
# for group in groups:
#     pyplot.subplot(len(groups), 1, i)
#     pyplot.plot(values[:, group])
#     pyplot.title(dataset.columns[group], y=0.5, loc='right')
#     i += 1
# pyplot.show()


values = dataset.loc[:, ~dataset.columns.isin(['vwap_d','vwap_v'])].values

# values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# reframed = series_to_supervised(values, 1, 1)


values = reframed.values
# Train Test split (80% train, 20% test)
train = values[:int(0.8*len(values))]
test = values[int(0.8*len(values)):]

X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

# y = df['vwap_v'].values
# y_train = y[:int(0.8*len(y))]
# y_test = y[int(0.8*len(y)):]


# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(X_train, y_train, epochs=40, batch_size=73, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.xlabel('epoch')
pyplot.ylabel('error')
pyplot.title('AAPL 10s VWAP')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, X_test[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test, X_test[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
print('R^2 score', r2_score(inv_y, inv_yhat))



pyplot.plot(y_test, label='actual vwap')
pyplot.plot(yhat, label='predicted vwap')
pyplot.xlabel('time (10s)')
pyplot.ylabel('normalized vwap')
pyplot.title('AAPL 10s VWAP')
pyplot.legend()
pyplot.show()





clf = linear_model.Lasso(alpha=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('mse',mean_squared_error(y_test, yhat))
print('R^2 score', r2_score(y_test, yhat))

# Random Forest
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=20, max_depth=3,random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))

# # SVM
# from sklearn.svm import LinearSVC
# clf = LinearSVC(random_state=0, tol=1e-5)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))


# # from scipy import stats
# # print(stats.itemfreq(y_test))