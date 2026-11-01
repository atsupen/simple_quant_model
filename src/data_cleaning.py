import typing
import yfinance as yf
import pandas as pd
from datetime import date

tsla = yf.Ticker("TSLA")
tesla_full = tsla.history(period = "max")
tesla_full.to_csv("data/tesla_prices_raw.csv", index = True)

tesla_clean = pd.read_csv("data/tesla_prices_raw.csv", parse_dates = ["Date"])

tesla_clean.columns = tesla_clean.columns.str.lower()
tesla_clean.columns = tesla_clean.columns.str.replace(" ", "_")
tesla_clean.date = tesla_clean.date.transform(lambda x : x[:10])

print(tesla_clean.info())

tesla_clean.to_csv("data/tesla_prices_clean.csv")