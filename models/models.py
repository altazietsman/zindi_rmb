import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import pandas as pd
import logging
from datetime import datetime
import os
import sys

# Import Statsmodels
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.statespace.sarimax import SARIMAX

import pmdarima as pm
from pmdarima.arima.utils import nsdiffs

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
            #np.asarray(y),
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        self.model_ = self.model_.fit()

        return self

    def predict(self, forecast=2):
        if self.model_ is None:
            raise RuntimeError("Model not trained")

        return self.model_.forecast(forecast).to_numpy()
    
class ProphetWrapper(BaseEstimator, RegressorMixin):

    def getModelName(self):
        return f"Prophet_{self.changepoint_range}_{self.n_changepoints}_{self.changepoint_prior_scale}_{self.name_postfix}"
    
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
    seasonality_mode='multiplicative', 
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
        self.n_changeponts=n_changepoints
        self.seasonality_mode=seasonality_mode

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

        df = df.dropna()

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

    def predict(self, forecast):
        if self.model_ is None:
            raise RuntimeError("Model not trained")

        future = self.model_.make_future_dataframe(periods=forecast, freq='MS')

        if type(self.extra_data) == pd.core.frame.DataFrame:
            df = self.extra_data.rename(columns={'date': 'ds'})
            df['ds'] = pd.DatetimeIndex(df['ds'])
            future = pd.merge(future, df, on='ds', how='left')

        predictions = self.model_.predict(future)


        return predictions['yhat'].tail(forecast).to_numpy()


class VarimaWrapper(BaseEstimator, RegressorMixin):

    def getModelName(self):
        return f"Varima_{self.name_postfix}"
    
    def getExtraData(self):
        return self.extra_data
    
    def __init__(self, extra_data=None, name_postfix=""):
        self.model_ = None,
        self.extra_data = extra_data
        self.name_postfix = name_postfix

    def fit(self, X, y):

        y.columns = ['index', 'y']
        y = y.reset_index()[['index', 'y']]

        data = pd.merge(y, X, right_on='date', left_on='index', how='left')

        data = data.dropna()

        #data['date'].map(lambda x: datetime.strptime(x, '%Y-%m'))
        #data['date'].max()

        prediction_date = pd.to_datetime(data['date'].max()) + pd.DateOffset(months=1)
        
        data = data.drop(['index'], axis=1).set_index('date')

        self.model_ = VAR(endog=data).fit()
        

    def predict(self, forecast):
        if self.model_ is None:
            raise RuntimeError("Model not trained")

        predictions = self.model_.forecast(self.model_.endog, steps=forecast)
        results = [prediction[0] for prediction in predictions]

        return results

class AutoArimaWrapper(BaseEstimator, RegressorMixin):

    def getModelName(self):
        return f"AutoArima_{self.name_postfix}"
    
    def getExtraData(self):
        return self.extra_data
    
    def __init__(self, extra_data=None, name_postfix=""):
        self.model_ = None,
        self.extra_data = extra_data
        self.name_postfix = name_postfix

    def fit(self, X, y):

        # The Arima model uses only one column for y
        y = y.iloc[:,1:]

        # Determine the number of seasonal differences using a Canova-Hansen test
        seasonal_diff = nsdiffs(y,
                m=12,  # commonly requires knowledge of dataset
                max_D=12,
                test='ch')  # -> 0

        # Fit the auto_arima model and find the best model
        self.model_ = pm.auto_arima(y, 
                        start_p=1, 
                        start_q=1,
                        test='adf',        # use adftest to find optimal 'd'
                        max_p=3, max_q=3,  # maximum p and q
                        m=12,              # frequency of series
                        d=None,            # let model determine 'd'
                        seasonal=True,     # Seasonality
                        start_P=1, 
                        D=seasonal_diff,
                        start_Q=1,
                        trace=False,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True,
                        disp=0)
        # Fit the model
        self.model_ = self.model_.fit(y, disp=0)


        
        
        

    def predict(self, forecast):
        if self.model_ is None:
            raise RuntimeError("Model not trained")

        forecast=self.model_.predict(n_periods=forecast, return_conf_int=True)
        return forecast[0].to_numpy()
