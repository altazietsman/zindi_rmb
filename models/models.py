import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

class HoltWintersWrapper(BaseEstimator, RegressorMixin):

    def getModelName(self):
        return "HoltWinters_{}_{}_{}".format(self.trend, self.seasonal, self.seasonal_periods)
    
    def __init__(self, trend="mul", seasonal="mul", seasonal_periods=12):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model_ = None

        # Set the random seed
        np.random.seed(111)

    def fit(self, X, y):

        # The holtswinter model uses only one column for y
        y = y.iloc[:,1:]

        self.model_ = ExponentialSmoothing(
            np.asarray(y),
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        self.model_ = self.model_.fit()

        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not trained")

        return self.model_.forecast(len(X))
    
class ProphetWrapper(BaseEstimator, RegressorMixin):

    def getModelName(self):
        return "Prophet"
    
    def __init__(self, interval_width=0.80, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False):
        self.interval_width = interval_width
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model_ = None

    def fit(self, X, y):

        if X != None:
            df = X.copy()
            df['y'] = y
        else:
            df = y.copy()
            #df = df.reset_index()
            df.columns = ['ds', 'y']

        self.model_ = Prophet(
            interval_width=self.interval_width,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        self.model_.fit(df)

        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not trained")

        future = self.model_.make_future_dataframe(periods=len(X))
        forecast = self.model_.predict(future)

        return forecast['yhat'].tail(len(X)).to_numpy()