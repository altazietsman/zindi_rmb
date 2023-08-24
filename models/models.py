import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import logging

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

    def fit(self, y, X=None):

        # The holtswinter model uses only one column for y
        # y = y.iloc[:,1:]

        self.model_ = ExponentialSmoothing(
            np.asarray(y),
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        self.model_ = self.model_.fit()

        return self

    def predict(self, forcast=2):
        if self.model_ is None:
            raise RuntimeError("Model not trained")

        return self.model_.forecast(forcast)
    
class ProphetWrapper(BaseEstimator, RegressorMixin):

    def getModelName(self):
        return "Prophet"
    
    def __init__(self, n_changepoints, seasonality_mode, interval_width=0.80):
        self.interval_width = interval_width
        self.model_ = None
        self.n_changeponts=n_changepoints
        self.seasonality_mode=seasonality_mode

    def fit(self, X, y=None):

        self.model_ = Prophet(interval_width=self.interval_width, seasonality_mode=self.seasonality_mode, n_changepoints=self.n_changeponts)

        regressors = X.columns
        for regressor in regressors:

            if (regressor == 'y') or (regressor == 'ds'):
                pass
            else:
                self.model_.add_regressor(regressor)

        logging.getLogger("cmdstanpy").disabled = True #  turn 'cmdstanpy' logs off
        self.model_.fit(X)
        logging.getLogger("cmdstanpy").disabled = False #  revert original setting

        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not trained")

        # future = self.model_.make_future_dataframe(periods=len(X))
        # for regressor in X.columns:
        #     future[regressor] = X[regressor]
        forecast = self.model_.predict(X)

        return forecast['yhat']