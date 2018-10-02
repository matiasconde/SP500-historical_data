import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

sphist = pd.read_csv("sphist.csv")

sphist["Date"] = pd.to_datetime(sphist["Date"])

sphist = sphist.sort_values(by="Date",ascending=True)

# Taking ratios to get features 

rolling_5 = sphist["Close"].rolling(5).mean()
rolling_365 = sphist["Close"].rolling(365).mean()
rolling_std365 = sphist["Close"].rolling(365).std()

rolling_std_volume5 = sphist["Volume"].rolling(5).std()
rolling_std_volume365 = sphist["Volume"].rolling(365).std()

rolling_volume5 = sphist["Volume"].rolling(5).mean()
rolling_volume365 = sphist["Volume"].rolling(365).mean()

rolling_5 = rolling_5.shift(periods=1)
rolling_365 = rolling_365.shift(periods=1)
rolling_std365 = rolling_std365.shift(periods=1)
rolling_std_volume365 = rolling_std_volume365.shift(periods=1)
rolling_std_volume5 = rolling_std_volume5.shift(periods=1)
rolling_volume5 = rolling_volume5.shift(periods=1)
rolling_volume365 = rolling_volume365.shift(periods=1)

print(rolling_5.head(8))
print(rolling_365.tail(8))
print(rolling_std365.tail(8))

sphist["last_5d_mean"] = rolling_5
sphist["last_365d_mean"] = rolling_365
sphist["last_365d_std"] = rolling_std365
sphist["past_5days_std_volume_over_last_year_std_volume"] = rolling_std_volume5 / rolling_std_volume365
sphist["past_5days_volume_over_last_year_volume"] = rolling_volume5 / rolling_volume365

sphist["last_5days_volume"] = rolling_volume5

print(sphist.head(8))
print(sphist.tail())

sphist.dropna(axis=0,inplace=True)

train = sphist[sphist["Date"] <= datetime(year=2013,month=1,day=1)]
test = sphist[sphist["Date"] > datetime(year=2013,month=1,day=1)]

print(train.head())
print(test.head())
print(train.columns)

features = ['last_5d_mean', 'last_365d_mean', 'last_365d_std']
features2 = ['last_5d_mean', 'last_365d_mean', 'last_365d_std',"past_5days_volume_over_last_year_volume"]
features3 = ['last_5d_mean', 'last_365d_mean', 'last_365d_std',"last_5days_volume","past_5days_volume_over_last_year_volume","past_5days_std_volume_over_last_year_std_volume"]

lr = LinearRegression()

lr.fit(train[features3],train["Close"])

predictions = lr.predict(test[features3])
predictions_series = pd.Series(name="Predicted_Prices",data=predictions)

prediction_error = mean_absolute_error(test["Close"],predictions)

test_Close = test["Close"].reset_index()

comparisons = pd.concat((predictions_series,test_Close),axis=1)

print(comparisons)
print(prediction_error)

# Accuracy would improve greatly by making predictions only one day ahead. This more closely simulates what you'd do if you were trading using the algorithm.

# Simulating one_day_ahead:
# 2015 -12 - 1,2,3,4,7 Last dates

train2 = sphist[sphist["Date"] <= datetime(year=2015,month=12,day=4)]
test2 = sphist[sphist["Date"] == datetime(year=2015,month=12,day=7)]

lr2 = LinearRegression()

lr2.fit(train2[features3],train2["Close"])

predictions2 = lr2.predict(test2[features3])
predictions_series2 = pd.Series(name="Predicted_Prices",data=predictions2)

prediction_error2 = mean_absolute_error(test2["Close"],predictions2)

test_Close2 = test2["Close"].reset_index()

comparisons2 = pd.concat((predictions_series2,test_Close2),axis=1)

print(comparisons2)
print(prediction_error2)

"""
CANDIDATES FOR RATIOS: 

- The average volume over the past five days.
- The average volume over the past year.
- The ratio between the average volume for the past five days, and the average volume for the past year.
- The standard deviation of the average volume over the past five days.
- The standard deviation of the average volume over the past year.
- The ratio between the standard deviation of the average volume for the past five days, and the standard deviation of the average volume for the past year.
- The year component of the date.
- The ratio between the lowest price in the past year and the current price.
- The ratio between the highest price in the past year and the current price.
- The year component of the date.
- The month component of the date.
- The day of week.
- The day component of the date.
- The number of holidays in the prior month.


SOME IDEAS TO CONTINUE:

We can improve the algorithm used significantly trying another techniques, like random forest.

We can incorporate outside data, such as the weather in New York City (where most trading happens) the day before, and the amount of Twitter activity around certain stocks.

We can make the system real-time by writing an automated script to download the latest data when the market closes, and make predictions for the next day.

Finally, we can make the system "higher-resolution". We're currently making daily predictions, but we could make hourly, minute-by-minute, or second by second predictions. This will require obtaining more data, though. We could also make predictions for individual stocks instead of the S&P500.
"""
