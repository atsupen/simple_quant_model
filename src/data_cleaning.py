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

initial_year, initial_mon, initial_day = [int(s) for s in tesla_clean.iloc[0, 0].split("-")]
initial_time = date(initial_year, initial_mon, initial_day)

def str2date(date_string):
    Y, M, D = [int(s) for s in date_string.split("-")]
    date_obj = date(Y, M, D)
    return date_obj

tesla_clean.date = tesla_clean.date.transform(lambda d : (str2date(d) - initial_time).days)
tesla_clean = tesla_clean.rename(columns = {"date" : "time"})

tesla_clean.to_csv("data/tesla_prices_clean.csv")