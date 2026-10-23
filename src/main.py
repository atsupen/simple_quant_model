import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

teslaPrices = pd.read_csv("data/tesla_prices_clean.csv")
teslaPrices = teslaPrices.drop(columns = "Unnamed: 0")