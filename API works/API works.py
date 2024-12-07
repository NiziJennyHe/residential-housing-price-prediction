#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import typing
import json
import datetime
import yfinance as yf
import statsmodels.api as sm
import itertools

class AVParameters(typing.TypedDict, total=False):
    """
    Valid query parameters for the AlphaVantage API.
    """

    adjusted: typing.Literal["true", "false"]
    apikey: typing.Required[str]
    datatype: typing.Literal["csv", "json"]
    extended_hours: typing.Literal["true", "false"]
    function: typing.Literal[
        "BALANCE_SHEET",
        "FEDERAL_FUNDS_RATE",
        "FEDERAL_FUNDS_RATE&maturity=1month"
        "TIME_SERIES_INTRADAY",
        "TIME_SERIES_DAILY",
        "TIME_SERIES_DAILY_ADJUSTED",
        "TIME_SERIES_WEEKLY",
        "TIME_SERIES_WEEKLY_ADJUSTED",
        "TIME_SERIES_MONTHLY",
        "TIME_SERIES_MONTHLY_ADJUSTED",
        "GLOBAL_QUOTE",
        "SYMBOL_SEARCH",
        "MARKET_STATUS",
        "INFLATION",
        "UNEMPLOYMENT"
    ]
    interval: typing.Literal[
        "1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"
    ]
    month: str
    symbol: str
    maturity: str
    


def get_alphavantage(params: AVParameters, use_cache: bool = True) -> typing.Any:
    """
    Makes an HTTP GET request to the AlphaVantage API, with caching.

    Parameters:
        params (AVParameters): The query parameters to include in the request
        use_cache (bool): Whether to use a cached response

    Returns:
        response (typing.Any): A JSON response
    """
    CACHE_DIR = "/tmp/fina4460"
    os.makedirs(CACHE_DIR, exist_ok=True)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    params_str = "_".join(
        f"{key}={value}" for key, value in sorted(params.items()) if key != "apikey"
    )
    cache_path = os.path.join(CACHE_DIR, f"{current_date}_{params_str}.json")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    response = requests.get(
        url="https://www.alphavantage.co/query?",
        params=typing.cast(typing.Mapping[str, typing.Any], params),
    )
    response.raise_for_status()
    data = response.json()
    if "Information" in data and "premium" in data["Information"]:
        if os.path.exists(cache_path):
            os.remove(cache_path)
        raise ValueError("AlphaVantage API rate limit hit. Please try again later.")
    if "Error Message" in data:
        if os.path.exists(cache_path):
            os.remove(cache_path)
        raise ValueError("Invalid call to AlphaVantage API.")
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data

def prices(apikey: str, ticker: str, window: int = 12) -> pd.DataFrame:
    """
    Gets historical prices for a specified ticker from the AlphaVantage API.

    Parameters:
        apikey (str): An AlphaVantage API key
        ticker (str): The stock ticker to get prices for
        window (int): The number of days to get prices for

    Returns:
        prices (pd.DataFrame): The historical prices
    """
    if window < 0:
        raise ValueError("window size must be positive")

    params: AVParameters = {
        "apikey": apikey,
        "datatype": "json",
        "function": "TIME_SERIES_MONTHLY_ADJUSTED",
        "symbol": ticker,
    }

    json_data = get_alphavantage(params = params)
    full_time_series = list(json_data["Monthly Adjusted Time Series"].items())
    windowed_time_series = full_time_series[:window][::-1]
    prices_list = [
        [
            pd.to_datetime(key),
            float(value["1. open"]) * adj_ratio,
            float(value["5. adjusted close"]),
            float(value["2. high"]) * adj_ratio,
            float(value["3. low"]) * adj_ratio,
            float(value["1. open"]),
            float(value["4. close"]),
            float(value["2. high"]),
            float(value["3. low"]),
        ]
        for key, value in windowed_time_series
        if (adj_ratio := (float(value["5. adjusted close"]) / float(value["4. close"])))
    ]
    prices = pd.DataFrame(
        prices_list,
        columns=[
            "date",
            "adj_open",
            "adj_close",
            "adj_high",
            "adj_low",
            "open",
            "close",
            "high",
            "low",
        ],
    )

    if any(
        column not in prices.columns
        for column in ["date", "adj_open", "adj_close", "adj_high", "adj_low"]
    ):
        raise ValueError("prices dataframe missing expected columns")
    return prices

def federal_funds_rates(apikey: str, window: int = 12 * 5) -> pd.DataFrame:
    """
    Gets the historical federal funds rates from the AlphaVantage API.

    Parameters:
        apikey (str): An AlphaVantage API key
        window (int): The number of months to get the federal funds rate for

    Returns:
        rates (pd.DataFrame): The historical federal funds rates
    """
    if window < 0:
        raise ValueError("window size must be positive")
        
    params: AVParameters = {
        "apikey": apikey,
        "datatype": "json",
        "function": "FEDERAL_FUNDS_RATE",
        "interval": "monthly",
    }

    json_data = get_alphavantage(params = params)
    windowed_time_series = json_data["data"][:window][::-1]
    
    rates = pd.DataFrame(windowed_time_series)
    rates["date"] = pd.to_datetime(rates["date"])
    rates.rename(columns={"value": "fed_funds"}, inplace=True)
    rates["fed_funds"] = pd.to_numeric(rates["fed_funds"])
    rates["fed_funds"] = rates["fed_funds"] / 100
    rates["fed_funds"] = (1 + rates["fed_funds"]) ** (1 /12) - 1

    if any(column not in rates.columns for column in ["date", "fed_funds"]):
        raise ValueError("rates dataframe missing expected columns")
    return rates

def unemployment(apikey: str, window: int = 12 * 5) -> pd.DataFrame:
    """
    Gets the historical unemployment rates from the AlphaVantage API.

    Parameters:
        apikey (str): An AlphaVantage API key
        window (int): The number of months to get the unemployment rate for

    Returns:
        rates (pd.DataFrame): The historical unemployment rates
    """
    if window < 0:
        raise ValueError("window size must be positive")
        
    params: AVParameters = {
        "apikey": apikey,
        "datatype": "json",
        "function": "UNEMPLOYMENT",
    }

    json_data = get_alphavantage(params = params)
    windowed_time_series = json_data["data"][:window][::-1]
    
    rates = pd.DataFrame(windowed_time_series)
    rates["date"] = pd.to_datetime(rates["date"])
    rates.rename(columns={"value": "unemployment"}, inplace=True)
    rates["unemployment"] = pd.to_numeric(rates["unemployment"])
    rates["unemployment"] = rates["unemployment"] / 100

    if any(column not in rates.columns for column in ["date", "unemployment"]):
        raise ValueError("rates dataframe missing expected columns")
    return rates

def inflation(apikey: str, window: int = 5) -> pd.DataFrame:
    """
    Gets the historical inflation rates from the AlphaVantage API.

    Parameters:
        apikey (str): An AlphaVantage API key
        window (int): The number of years to get the inflation rates for

    Returns:
        rates (pd.DataFrame): The historical inflation rates
    """
    if window < 0:
        raise ValueError("window size must be positive")
        
    params: AVParameters = {
        "apikey": apikey,
        "datatype": "json",
        "function": "INFLATION",
    }

    json_data = get_alphavantage(params = params)
    windowed_time_series = json_data["data"][:window][::-1]
    
    rates = pd.DataFrame(windowed_time_series)
    rates["date"] = pd.to_datetime(rates["date"])
    rates.rename(columns={"value": "inflation"}, inplace=True)
    rates["inflation"] = pd.to_numeric(rates["inflation"])
    rates["inflation"] = rates["inflation"] / 100

    if any(column not in rates.columns for column in ["date", "inflation"]):
        raise ValueError("rates dataframe missing expected columns")
    return rates

if __name__ == "__main__":
    #key
    alphavantage_api_key = os.getenv("av_api_key")
    
    #Time horizon
    years = 5
    
    #Monthly S&P 500 (SPY)
    spy = prices(alphavantage_api_key, "SPY", 12 * years)
    print(spy)
    
    #1-month Fed Funds Rate
    fed_funds = federal_funds_rates(alphavantage_api_key)
    print(fed_funds)
    
    #Unemployment rate
    unemployment = unemployment(alphavantage_api_key)
    print(unemployment)
    
    #Inflation
    inflation = inflation(alphavantage_api_key)
    print(inflation)
    
    
    
    
    

