import toml
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os

import multiprocessing
import os
import random
from fredapi import Fred

script_path = os.path.dirname(os.path.realpath(__file__))
KEYS = toml.load(f"{script_path}/keys.toml")


class StockData(multiprocessing.Process):
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
        whether to print url links or not etc

    Methods
    -------
    """

    def __init__(self, ticker, verbose=0):
        """
        inherit from multiprocessing
        init variables
        instantiate the earnings / reporting calendar for this stock
        """
        multiprocessing.Process.__init__(self)
        self.ticker = ticker
        self.api_key = KEYS["fmp_apikey"]
        self.verbose = verbose
        self.get_calendar_data()

    def create_featureset(self):
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
        return

    def fmp_datapull(self, url, nested=False):
        """
        generic method to process return data from fmp
        """
        if nested == False:
            json = requests.get(url).json()
            df = pd.DataFrame(json)
        else:  # some links return data in a dictionary form
            json = requests.get(url).json()
            # ticker = list(json.keys())[0]
            data = list(json.keys())[1]
            df = pd.DataFrame(json[data])
        return df

    def get_calendar_data(self):
        url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{self.ticker}?apikey={self.api_key}"
        self.calendar = self.fmp_datapull(url)
        return

    def get_financial_data(self):
        """
        Income statement and balance sheet data
        """
        income_statement_url = f"https://financialmodelingprep.com/api/v3/income-statement-as-reported/{self.ticker}?period=quarterly&apikey={self.api_key}&limit=50"
        balance_sheet_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement-as-reported/{self.ticker}?period=quarter&apikey={self.api_key}&limit=50"
        self.income_statement = self.fmp_datapull(income_statement_url)
        self.balance_sheet = self.fmp_datapull(balance_sheet_url)
        return

    def get_price_data(self):
        """
        Price data historical
        """
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{self.ticker}?apikey={self.api_key}"
        self.price_data = self.fmp_datapull(url, nested=True)
        return

    def get_analyst_estimates(self):
        """
        Analyst estimate data
        """
        url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{self.ticker}?apikey={self.api_key}"
        self.analyst_estimates = self.fmp_datapull(url)
        return

    def get_ratios(self):
        """
        Ratios data
        """
        url = f"https://financialmodelingprep.com/api/v3/ratios/{self.ticker}?apikey={self.api_key}&period=quarter"
        self.ratios = self.fmp_datapull(url)
        return

    def get_key_metrics(self):
        """
        Metrics data
        """
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{self.ticker}?apikey={self.api_key}&period=quarter"
        self.key_metrics = self.fmp_datapull(url)
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
