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


class AutoArima(BaseEstimator, RegressorMixin):
    """Wrapper class for the Auto Arima model

    Attributes:
    -----------
    extra_data: pandas dataframe

    name_postfix: str
                  postfix that can be added to model name

    """

    def __init__(self, extra_data=None, name_postfix=""):
        self.model_ = (None,)
        self.name_postfix = name_postfix
        self.extra_data = extra_data

    def getModelName(self):
        """Method to retrieve name of model"""
        return f"AutoArima_{self.name_postfix}"

    def getExtraData(self):
        """Method to retrieve addition data added. This is not used fot holtwinters, but keeps model API standard across models"""
        return self.extra_data

    def fit(self, y, X=None):
        """Fits auto arima model

        X: not used in holtwinters
        y: pandas dataframe
           Data used for training if X is not provided. Needs to include predicted variable and data
        """

        # The Arima model uses only one column for y
        y = y.iloc[:, 1:]

        # Determine the number of seasonal differences using a Canova-Hansen test
        seasonal_diff = nsdiffs(
            y, m=12, max_D=12, test="ch"  # commonly requires knowledge of dataset
        )  # -> 0

        # Fit the auto_arima model and find the best model
        self.model_ = pm.auto_arima(
            y,
            start_p=1,
            start_q=1,
            test="adf",  # use adftest to find optimal 'd'
            max_p=6,
            max_q=6,  # maximum p and q
            m=12,  # frequency of series
            d=None,  # let model determine 'd'
            seasonal=True,  # Seasonality
            start_P=1,
            D=seasonal_diff,
            start_Q=1,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            disp=0,
            random_state=345,
            information_criterion="aic",
        )
        # Fit the model
        self.model_ = self.model_.fit(y, disp=0)

        return self

    def predict(self, forecast=1):
        """Makes prediction with fitted model

        Arguments:
        ----------
        forecast: int
                  period to forcast on. Defaults to one month

        Returns:
        --------
        predictions: float
        """

        if self.model_ is None:
            raise RuntimeError("Model not trained")

        forecast = self.model_.predict(n_periods=forecast, return_conf_int=True)
        return forecast[0].to_numpy()
