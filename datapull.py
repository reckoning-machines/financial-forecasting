import toml
import warnings
import pandas as pd

pd.set_option("mode.chained_assignment", None)
import numpy as np
import requests
from datetime import datetime
import os

import multiprocessing
import os
import random
from fredapi import Fred
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from pmdarima.metrics import smape
import numpy as np
import matplotlib.pyplot as plt


script_path = os.path.dirname(os.path.realpath(__file__))
KEYS = toml.load(f"{script_path}/keys.toml")


class Forecast:
    """
    a class to handle a single time series forecast, either univariate or multivariate
    """

    def __init__(self, ticker="", forecast_column="", data_columns=[], data=None):
        self.data = data
        self.ticker = ticker
        self.target_column = forecast_column
        self.data_columns = data_columns
        self.forecast_periods = 1

    def fit(self, forecast_column="value", feature_columns=[], seasonal=False, m=4):
        # Load/split your data
        g = self.data.dropna()
        if feature_columns != []:
            exog = feature_columns
            data = g.dropna()
            g[exog] = g[exog].astype(float)

            model = pm.auto_arima(
                # maxiter=30,
                # method="nm",
                g[forecast_column],
                g[exog],
                m=m,
                seasonal=seasonal,
                test="adf",
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                trace=False,
                method="nm",
                maxiter=30,
            )
            model.fit(g[forecast_column], g[exog])
            fitted, conf_int = model.predict_in_sample(
                return_conf_int=True, alpha=0.05, X=g[exog]
            )
        else:
            model = pm.auto_arima(
                g[forecast_column],
                m=m,
                seasonal=seasonal,
                test="adf",
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                trace=False,
            )
            model.fit(g[forecast_column])
            fitted, conf_int = model.predict_in_sample(return_conf_int=True, alpha=0.05)

        g["predictions"] = fitted
        conf_int = pd.DataFrame(conf_int)

        conf_int.columns = ["lower_confidence", "upper_confidence"]
        g["lower_confidence"] = conf_int["lower_confidence"].values
        g["upper_confidence"] = conf_int["upper_confidence"].values

        g["lower_confidence"] = np.where(
            g["lower_confidence"] < -100, 0, g["lower_confidence"]
        )
        g["lower_confidence"] = np.where(
            g["lower_confidence"] > 100, 0, g["lower_confidence"]
        )
        g["upper_confidence"] = np.where(
            g["upper_confidence"] < -100, 0, g["upper_confidence"]
        )
        g["upper_confidence"] = np.where(
            g["upper_confidence"] > 100, 0, g["upper_confidence"]
        )

        x = model.summary()
        d = pd.DataFrame(x.tables[1])[[0, 1]]
        d = d.iloc[1:, :].T
        d.columns = d.iloc[0]
        d = d[1:]
        d.columns = d.columns.astype(str)

        for col in feature_columns:
            g[col + "_coef"] = float(d[col].iloc[0].data)
        self.mape = (
            (g[forecast_column] - g.predictions).abs() / g[forecast_column]
        ).mean()
        self.correlation = g[forecast_column].corr(g["predictions"])

        corr_matrix = np.corrcoef(g[forecast_column], g.predictions)
        corr = corr_matrix[0, 1]
        self.r_squared = corr**2

        self.model = model
        self.forecast_data = g.tail(-2)
        return


class Featureset(multiprocessing.Process):
    """
    A class to represent a stock.
    Inherits from multi processing to facilitate use of cores
    for instantiating several of these classes in a list
    ...

    Attributes
    ----------
    ticker : str
        stock ticker ... no exchange
        Example: AAPL
    verbose : bool
        whether to print url links or not etc (unused)

    Methods
    -------
    """

    def __init__(self, ticker, verbose=0):
        """
        inherit from multiprocessing
        init variables
        instantiate the data dictionary for available datapulls (in this case fmp)
        instantiate the earnings / reporting calendar for this stock
        """
        multiprocessing.Process.__init__(self)
        self.ticker = ticker
        self.api_key = KEYS["fmp_apikey"]
        self.verbose = verbose
        self.data_dictionary = {
            "calendar": {
                "url": f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{self.ticker}?apikey={self.api_key}",
                "nested": False,
            },
            "income_statement": {
                "url": f"https://financialmodelingprep.com/api/v3/income-statement-as-reported/{self.ticker}?period=quarterly&apikey={self.api_key}&limit=50",
                "nested": False,
            },
            "prices": {
                "url": f"https://financialmodelingprep.com/api/v3/historical-price-full/{self.ticker}?apikey={self.api_key}",
                "nested": True,
            },
            "analyst_estimates": {
                "url": f"https://financialmodelingprep.com/api/v3/analyst-estimates/{self.ticker}?apikey={self.api_key}",
                "nested": False,
            },
            "ratios": {
                "url": f"https://financialmodelingprep.com/api/v3/ratios/{self.ticker}?apikey={self.api_key}&period=quarter",
                "nested": False,
            },
            "key_metrics": {
                "url": f"https://financialmodelingprep.com/api/v3/key-metrics/{self.ticker}?apikey={self.api_key}&period=quarter",
                "nested": False,
            },
        }
        self.fmp_datapull("calendar")

    def fmp_datapull(self, data_key):
        """
        generic method to process return data from fmp
        BUT this could be a kusto call or reddis etc
        """
        data_dictionary = self.data_dictionary[data_key]
        if data_dictionary["nested"] == False:
            json = requests.get(data_dictionary["url"]).json()
            df = pd.DataFrame(json)
        else:  # some links return data in a dictionary form
            json = requests.get(data_dictionary["url"]).json()
            # ticker = list(json.keys())[0]
            data = list(json.keys())[1]
            df = pd.DataFrame(json[data])
        if "date" in df.columns:
            df = df.sort_values("date")
        setattr(self, data_key, df)
        return df

    def join(self,data_key):
        """
        generic method to left join data to calendar dataframe
        """

        if not hasattr(self, 'featureset'):
            """
            instantiate the calendar as the 'binder'
            presumes calendar exists
            """
            self.featureset = self.calendar.sort_values("date")
            self.featureset["date"] = pd.to_datetime(self.featureset["date"])
            self.featureset["report_date"] = pd.to_datetime(self.featureset["date"])
            self.featureset["previous_report_date"] = pd.to_datetime(
                self.featureset["date"]
            ).shift(1)

            self.featureset = self.featureset.loc[~self.featureset["epsEstimated"].isna()]
            self.featureset = (
                self.featureset.set_index("date").resample("D").ffill().reset_index()
            )
        df = getattr(self,data_key)
        self.featureset = self.featureset.merge(df,how='left',left_on='date',right_on='date')
        return



from datetime import date


class FredData(multiprocessing.Process):
    """
    A class to represent a single data series from the st louis fed.
    Inherits from multi processing to facilitate use of cores
    for instantiating several of these classes in a list
    ...

    Attributes
    ----------
    ticker : str
        FRED series
        Example:LAUST060000000000003A
    verbose : bool
        whether to print url links or not etc

    Methods
    -------
    """

    def __init__(self, ticker, verbose=0):
        multiprocessing.Process.__init__(self)
        self.ticker = ticker
        self.api_key = KEYS["fred_apikey"]
        self.verbose = verbose

    def get_data(self):
        FRED = Fred(self.api_key)
        today = date.today()
        today = today.strftime("%Y-%m-%d")

        df_fred = pd.DataFrame()
        s = pd.DataFrame(
            FRED.get_series(
                self.ticker,
                observation_start="2017-01-01",
                observation_end=today,
            )
        ).reset_index()
        s.columns = ["end_date", self.ticker]
        s[self.ticker] = s[self.ticker] / 100
        s = s.groupby("end_date").last().reset_index()
        if df_fred.shape[0] > 0:
            # s = s.drop(columns="end_date")
            df_fred = pd.merge(
                df_fred,
                s,
                how="left",
                left_on="end_date",
                right_on="end_date",
            )
        else:
            df_fred = s
        df_fred["date"] = pd.to_datetime(df_fred["end_date"])

        df_fred = df_fred.ffill()
        df_fred = df_fred.set_index("date").resample("D").mean().reset_index()
        df_fred["treasury_calc"] = "mean"
        df_fred["quarter"] = df_fred["date"].dt.quarter.astype(str)
        df_fred["month"] = df_fred["date"].dt.month.astype(str)
        df_fred["year"] = df_fred["date"].dt.year.astype(str)
        self.data = df_fred
        return
