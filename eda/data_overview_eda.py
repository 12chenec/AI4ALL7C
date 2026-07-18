# Import Packages
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates

# Load the merged datasets
rolling_data = pd.read_csv("model_ready_weekly.csv")
era_data = pd.read_csv("model_ready_era2022.csv")

def print_column_names(df):
    """
    Takes a pandas DataFrame and prints all of its column names.
    """
    for column in df.columns:
        print(column)

def print_df_shape(df):
    """
    Prints the number of rows and columns in a pandas DataFrame.
    """
    rows, cols = df.shape
    print(f"Dataset Dimensions: {rows:,} rows and {cols} columns.")

def plot_distributions(df):
    """
    Plots all the columns in histograms.
    """
    columns = [
        "log10_conc_mean",
        "log10_conc_median",
        "conc_site_z_mean",
        "conc_delta_1w",
        "conc_roll3",
        "admits_this_week",
        "admits_per100k",
        "n_samples",
        "n_sites",
        "pct_nondetect",
        "pop_served",
        "coverage",
        "regime",
        "pct_change_next",
        "y_reg_next_admits"
        # "y_surge_next_week"
    ]

    for col in columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col].dropna(), bins="fd", edgecolor="black")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

def plot_distributions(df1, df2, save_dir):
    """
    Plots side-by-side histograms of the same columns from two datasets
    and saves each figure.

    Parameters:
        df1 (pd.DataFrame): First dataset.
        df2 (pd.DataFrame): Second dataset.
        save_dir (str): Directory to save histogram images.
    """

    os.makedirs(save_dir, exist_ok=True)

    columns = [
        "log10_conc_mean",
        "log10_conc_median",
        "conc_site_z_mean",
        "conc_delta_1w",
        "conc_roll3",
        "admits_this_week",
        "admits_per100k",
        "n_samples",
        "n_sites",
        "pct_nondetect",
        "pop_served",
        "coverage",
        # "regime",
        "pct_change_next",
        "y_reg_next_admits"
        # "y_surge_next_week"
    ]

    for col in columns:
        data1 = df1[col].dropna()
        data2 = df2[col].dropna()

        # Use common bin edges for fair comparison
        combined = np.concatenate([data1, data2])
        bins = np.histogram_bin_edges(combined, bins="fd")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        axes[0].hist(data1, bins=bins, edgecolor="black")
        axes[0].set_title(f"Dataset 1: {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")

        axes[1].hist(data2, bins=bins, edgecolor="black")
        axes[1].set_title(f"Dataset 2: {col}")
        axes[1].set_xlabel(col)

        plt.suptitle(f"Distribution of {col}")
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, f"{col}_distribution.png"), dpi=300)
        plt.close(fig)

# print_column_names(era_data)
# print_df_shape(era_data)

# plot_distributions(era_data)
# plot_distributions(rolling_data, era_data, "eda_figures/distributions")