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


class ProphetWrapper(BaseEstimator, RegressorMixin):
    """Wrapper class for the prophet model

    Attributes:
    -----------
    interval_width: float
    yearly_seasonality: str
    weekly_seasonality: boolean
    daily_seasonality: boolean
    changepoint_prior_scale: float
    n_changepoints: int
    changepoint_range: float
    name_postfix: str
                  postfix that can be added to model name
    seasonality_mode: str
    extra_data: pandas dataframe
                additional data that can be added to model for predictions

    """

    def __init__(
        self,
        interval_width=0.80,
        yearly_seasonality="auto",
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        n_changepoints=20,
        changepoint_range=0.8,
        name_postfix="",
        seasonality_mode="multiplicative",
        extra_data=None,
    ):
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
        self.n_changeponts = n_changepoints
        self.seasonality_mode = seasonality_mode

    def getModelName(self):
        """Method to retrieve name of model"""
        return f"Prophet_{self.changepoint_range}_{self.n_changepoints}_{self.changepoint_prior_scale}_{self.name_postfix}"

    def getExtraData(self):
        """Method to retrieve addition data added"""
        return self.extra_data

    def fit(self, X, y):
        """Fits prophet model. If X data is provided, this data is used for training, otherwise the y data is used as data.

        Arguments:
        ----------
        X: pandas dataframe
           Data used for training. Needs to include predicted variable and data, however can include additional
           trend data to be included as well.
        y: pandas dataframe
           Data used for training if X is not provided. Needs to include predicted variable and data
        """

        if type(X) == pd.core.frame.DataFrame:
            df = y.copy()
            df.columns = ["date", "y"]
            df = pd.merge(df, X, on="date", how="inner")
            df.rename(columns={"date": "ds"}, inplace=True)
        else:
            df = y.copy()
            # df = df.reset_index()
            df.columns = ["ds", "y"]

        df["ds"] = pd.DatetimeIndex(df["ds"])

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
            if col != "ds" and col != "y":
                self.model_.add_regressor(col)

        logging.getLogger("cmdstanpy").disabled = True  #  turn 'cmdstanpy' logs off
        self.model_.fit(df)
        logging.getLogger("cmdstanpy").disabled = False  #  revert original setting

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

        future = self.model_.make_future_dataframe(periods=forecast, freq="MS")

        if type(self.extra_data) == pd.core.frame.DataFrame:
            df = self.extra_data.rename(columns={"date": "ds"})
            df["ds"] = pd.DatetimeIndex(df["ds"])
            future = pd.merge(future, df, on="ds", how="left")

        predictions = self.model_.predict(future)

        return predictions["yhat"].tail(forecast).to_numpy()
