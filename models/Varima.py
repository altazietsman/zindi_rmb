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


class Varima(BaseEstimator, RegressorMixin):
    """Wrapper class for the prophet model

    Attributes:
    -----------
    name_postfix: str
                  postfix that can be added to model name
    extra_data: pandas dataframe
                additional data that can be added to model for predictions

    """

    def __init__(self, extra_data=None, name_postfix=""):
        self.model_ = (None,)
        self.extra_data = extra_data
        self.name_postfix = name_postfix

    def getModelName(self):
        """Method to retrieve name of model"""
        return f"Varima_{self.name_postfix}"

    def getExtraData(self):
        """Method to retrieve addition data added"""
        return self.extra_data

    def fit(self, y, X=None):
        """Fits prophet model. If X data is provided, this data is used for training, otherwise the y data is used as data.

        Arguments:
        ----------
        X: pandas dataframe
           Additional data used for training.
           trend data to be included as well.
        y: pandas dataframe
           Data for trand to predict. Needs to include predicted variable and date
        """

        y.columns = ["index", "y"]
        y = y.reset_index()[["index", "y"]]

        if type(X) == pd.core.frame.DataFrame:
            data = pd.merge(y, X, right_on="date", left_on="index", how="left")
            data = data.dropna()
            prediction_date = pd.to_datetime(data["date"].max()) + pd.DateOffset(months=1)
            data = data.drop(["index"], axis=1).set_index("date")
        else:
            data = y
            prediction_date = pd.to_datetime(data["index"].max()) + pd.DateOffset(months=1)
            data = data.set_index("index")

        self.model_ = VAR(endog=data).fit()

        return self.model_

    def predict(self, forecast):
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

        predictions = self.model_.forecast(self.model_.endog, steps=forecast)
        results = [prediction[0] for prediction in predictions]

        return results
