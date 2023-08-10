import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import pandas as pd
import logging

class HoltWintersWrapper(BaseEstimator, RegressorMixin):

    def getModelName(self):
        return "HoltWinters_{}_{}_{}".format(self.trend, self.seasonal, self.seasonal_periods)
    
    def getExtraData(self):
        return None
    
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
        return f"Prophet_{self.changepoint_range}_{self.n_changepoints}_{self.changepoint_prior_scale}{self.name_postfix}"
    
    def getExtraData(self):
        return self.extra_data
    
    def __init__(self, 
    interval_width=0.80, 
    yearly_seasonality='auto', 
    weekly_seasonality=False, 
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    n_changepoints=20,
    changepoint_range=0.8, 
    name_postfix="", 
    extra_data=None):
        self.interval_width = interval_width
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.name_postfix = name_postfix
        self.extra_data = extra_data
        self.changepoint_prior_scale = changepoint_prior_scale
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.model_ = None

    def fit(self, X, y):

        if type(X) == pd.core.frame.DataFrame:
            df = y.copy()
            df.columns = ['date', 'y']
            df = pd.merge(df, X, on='date', how='inner')
            df.rename(columns={'date': 'ds'}, inplace=True)
        else:
            df = y.copy()
            #df = df.reset_index()
            df.columns = ['ds', 'y']

        df['ds'] = pd.DatetimeIndex(df['ds'])

        self.model_ = Prophet(
            interval_width=self.interval_width,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,

        )

        for col in df.columns:
            if col != 'ds' and col != 'y':
                self.model_.add_regressor(col)

        logging.getLogger("cmdstanpy").disabled = True #  turn 'cmdstanpy' logs off
        self.model_.fit(df)
        logging.getLogger("cmdstanpy").disabled = False #  revert original setting

        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not trained")

        future = self.model_.make_future_dataframe(periods=len(X), freq='MS')

        if type(self.extra_data) == pd.core.frame.DataFrame:
            df = self.extra_data.rename(columns={'date': 'ds'})
            df['ds'] = pd.DatetimeIndex(df['ds'])
            future = pd.merge(future, df, on='ds', how='left')

        forecast = self.model_.predict(future)

        return forecast['yhat'].tail(len(X)).to_numpy()