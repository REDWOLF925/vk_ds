import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tqdm import tqdm

def none_count(lst: list) -> int:
    return lst.count(None)

def convert_to_tsfresh_format(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.repeat(df["id"].values, df["dates"].str.len()),
            "label": np.repeat(df["label"].values, df["dates"].str.len()),
            "date": np.concatenate(df["dates"].values),
            "value": np.concatenate(df["values"].values),
        }
    )

def show_n_samples(df: pd.DataFrame, n: int) -> None:
    sampled_df = pd.concat([df.loc[df['label'] == 0].sample(n).reset_index(drop=True),
                            df.loc[df['label'] == 1].sample(n).reset_index(drop=True)
                            ])

    fig, axes = plt.subplots(1, 4, figsize=(25, 10))

    for i in range(2 * n):
        row = sampled_df.iloc[i]
        dates = pd.to_datetime(row["dates"])
        values = row["values"]
        label = row["label"]

        axes[0].set_title(f"Values")
        axes[0].plot(dates, values, color="green" if label else "orange", alpha=0.2 if label else 0.8)
        axes[0].tick_params(axis="x", rotation=30)

        decomposition = seasonal_decompose(
            values, model="additive", period=12, extrapolate_trend="freq"
        )
        axes[1].set_title("Trend")
        axes[1].plot(dates, decomposition.trend, color="green" if label else "orange", alpha=0.2 if label else 0.8)
        axes[1].tick_params(axis="x", rotation=30)
        axes[2].set_title("Seasonality")
        axes[2].plot(dates, decomposition.seasonal, color="green" if label else "orange", alpha=0.2 if label else 0.8)
        axes[2].tick_params(axis="x", rotation=30)

        axes[3].set_title("Power spectral density")
        freq, power = sp.signal.periodogram(values)
        axes[3].plot(freq, power, color="green" if label else "orange", alpha=0.2 if label else 0.8)

    plt.tight_layout()
    plt.show()

def show_ts_samples(df: pd.DataFrame) -> None:
    """Sample 6 random time series (3 of each class) and plot them on a 4x4 grid."""
    sampled_df = (
        df.groupby("label")
        .apply(lambda x: x.sample(3))
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(6, 4, figsize=(20, 20))

    for i, ax_row in enumerate(axes):
        row = sampled_df.iloc[i]
        dates = pd.to_datetime(row["dates"])
        values = row["values"]
        label = row["label"]

        # Plot timeseries itself
        ax_row[0].set_title("Time Series")
        ax_row[0].plot(dates, values, color="green" if label else "orange")
        ax_row[0].tick_params(axis="x", rotation=30)
        ax_row[0].legend([f"label: {label}"], loc="upper right")

        # Plot a trend and seasonality.
        decomposition = seasonal_decompose(
            values, model="additive", period=12, extrapolate_trend="freq"
        )
        ax_row[1].set_title("Trend")
        ax_row[1].plot(dates, decomposition.trend)
        ax_row[1].tick_params(axis="x", rotation=30)
        ax_row[2].set_title("Seasonality")
        ax_row[2].plot(dates, decomposition.seasonal)
        ax_row[2].tick_params(axis="x", rotation=30)

        # Plot the spectral decomposition
        ax_row[3].set_title("Power spectral density")
        freq, power = sp.signal.periodogram(values)
        ax_row[3].plot(freq, power)

    fig.suptitle("Time series examples from the dataset")
    plt.tight_layout()
    plt.show()

def max_ignore_none(lst):
    return max(filter(lambda x: x is not None, lst), default=None)
def min_ignore_none(lst):
    return min(filter(lambda x: x is not None, lst), default=None)
def none_count(lst):
    return lst.count(None)
def inflow_sum(lst):
    s = 0
    for i in range(1, len(lst)):
        diff = lst[i] - lst[i - 1]
        if diff > 0:
            s += diff
    return s
def outflow_sum(lst):
    s = 0
    for i in range(1, len(lst)):
        diff = lst[i] - lst[i - 1]
        if diff < 0:
            s -= diff
    return s

def inflow_len(lst):
    ans = 0
    s = 0
    for i in range(1, len(lst)):
        diff = lst[i] - lst[i - 1]
        if diff > 0:
            s += 1
            ans = max(ans, s)
        else:
            s = 0
    return ans
def outflow_len(lst):
    ans = 0
    s = 0
    for i in range(1, len(lst)):
        diff = lst[i] - lst[i - 1]
        if diff < 0:
            s += 1
            ans = max(ans, s)
        else:
            s = 0
    return ans

def inflow_max(lst):
    ans = 0
    for i in range(1, len(lst)):
        diff = lst[i] - lst[i - 1]
        if diff > 0:
            ans = max(ans, diff)
    return ans
def outflow_max(lst):
    ans = 0
    for i in range(1, len(lst)):
        diff = lst[i] - lst[i - 1]
        if diff < 0:
            ans = max(ans, -diff)
    return ans

def inflow_max_val(lst):
    ans = 0
    s = 0
    for i in range(1, len(lst)):
        diff = lst[i] - lst[i - 1]
        if diff > 0:
            s += diff
            ans = max(ans, s)
        else:
            s = 0
    return ans
def outflow_max_val(lst):
    ans = 0
    s = 0
    for i in range(1, len(lst)):
        diff = lst[i] - lst[i - 1]
        if diff < 0:
            s -= diff
            ans = max(ans, s)
        else:
            s = 0
    return ans

def static_count(lst):
    ans = 0
    eps = 10**(0)
    for i in range(1, len(lst)):
        diff = lst[i] - lst[i - 1]
        if abs(diff) <= eps:
            ans +=1
    return ans

def count_static(lst):
    s = 0
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
            s += 1
    return s

def mapping(x, a, b, c, d): 
    return a + b * x + c * x**2 + d * x**3

def create_new_features(df):
    df['Len'] = df['values'].apply(len)
    df['Max'] = df['values'].apply(max)
    df['Mean'] = df['values'].apply(np.mean)
    df['Sum'] = df['values'].apply(sum)
    df['Var'] = df['values'].apply(np.var)
    df['Inflow'] = df['values'].apply(inflow_sum)
    df['Outflow'] = df['values'].apply(outflow_sum)
    df['Static_count'] = df['values'].apply(static_count)

    coef = [0] * df.shape[0]
    bias = [0] * df.shape[0]
    a = [0] * df.shape[0]
    b = [0] * df.shape[0]
    c = [0] * df.shape[0]
    d = [0] * df.shape[0]

    ind = 0
    for index, row in tqdm(df.iterrows(), maxinterval=df.shape[0]):
        lr = LinearRegression()
        lr.fit(np.arange(len(row['values'])).reshape(-1, 1), row['values'])
        coef[ind] = lr.coef_[0]
        bias[ind] = lr.intercept_
        args, covar = curve_fit(mapping, np.arange(len(row['values'])), row['values'], maxfev = 100000)
        a[ind] = args[0]
        b[ind] = args[1]
        c[ind] = args[2]
        d[ind] = args[3]
        ind += 1

    df['Trend'] = coef
    df['Bias'] = bias

    df['A'] = a
    df['D'] = d
    return df