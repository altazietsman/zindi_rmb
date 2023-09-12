import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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


class HoltWinters(BaseEstimator, RegressorMixin):
    """Wrapper class for the holtwinters model

    Attributes:
    -----------
    trend: str
    seasonal: str
    seasonal_periods: int

    """

    def __init__(self, trend="mul", seasonal="mul", seasonal_periods=12):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model_ = None

        # Set the random seed
        np.random.seed(111)

    def getModelName(self):
        """Method to retrieve name of model"""
        return "HoltWinters_{}_{}_{}".format(
            self.trend, self.seasonal, self.seasonal_periods
        )

    def getExtraData(self):
        """Method to retrieve addition data added. This is not used fot holtwinters, but keeps model API standard across models"""
        return None

    def fit(self, y,  X=None):
        """Fits holtwinters model

        X: not used in holtwinters
        y: pandas dataframe
           Data used for training if X is not provided. Needs to include predicted variable and data
        """
        # The holtswinter model uses only one column for y
        y = y.iloc[:, 1:]

        self.model_ = ExponentialSmoothing(
            # np.asarray(y),
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        self.model_ = self.model_.fit()

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

        return self.model_.forecast(forecast).to_numpy()
