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

def plot_trends(df, state=None):
    """
    Plot normalized wastewater concentration and hospital admissions over time.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        state (str, optional): State abbreviation to filter by.
    """

    data = df.copy()

    if state is not None:
        data = data[data["state_territory"] == state]

    data["week_end"] = pd.to_datetime(data["week_end"])
    data = data.sort_values("week_end")

    scaler = MinMaxScaler()

    data["conc_scaled"] = scaler.fit_transform(
        data[["log10_conc_mean"]]
    )

    data["admits_scaled"] = scaler.fit_transform(
        data[["admits_this_week"]]
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        data["week_end"],
        data["conc_scaled"],
        label="Wastewater (log10_conc_mean)",
        linewidth=2,
    )

    ax.plot(
        data["week_end"],
        data["admits_scaled"],
        label="Hospital Admissions",
        linewidth=2,
    )

    # Show one tick per year
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.xlabel("Year")
    plt.ylabel("Normalized Value")
    plt.title(
        "Wastewater vs Hospital Admissions"
        + (f" ({state})" if state else "")
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_trends_2023(df, state=None):
    """
    Plot normalized wastewater concentration and hospital admissions
    for only the year 2023.
    """

    data = df.copy()

    if state is not None:
        data = data[data["state_territory"] == state]

    data["week_end"] = pd.to_datetime(data["week_end"])
    data = data.sort_values("week_end")

    # Keep only 2023
    data = data[
        (data["week_end"] >= "2023-01-01") &
        (data["week_end"] < "2024-01-01")
    ]

    if data.empty:
        print("No data available for 2023.")
        return

    scaler = MinMaxScaler()

    data["conc_scaled"] = scaler.fit_transform(
        data[["log10_conc_mean"]]
    )

    data["admits_scaled"] = scaler.fit_transform(
        data[["admits_this_week"]]
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        data["week_end"],
        data["conc_scaled"],
        linewidth=2,
        label="Wastewater (log10_conc_mean)"
    )

    ax.plot(
        data["week_end"],
        data["admits_scaled"],
        linewidth=2,
        label="Hospital Admissions"
    )

    # Show one tick per month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    plt.xlabel("2023")
    plt.ylabel("Normalized Value")
    plt.title(
        "Wastewater vs Hospital Admissions (2023)"
        + (f" - {state}" if state else "")
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_zscore_vs_next_admits(df, state=None, year=None):
    """
    Plot conc_site_z_mean and next-week hospital admissions over time.

    Both variables are Min-Max scaled to 0–1 so their trends are comparable.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        state (str, optional): State abbreviation or name to filter by.
        year (int, optional): Year to display, such as 2023.
    """

    data = df.copy()

    required_columns = [
        "week_end",
        "conc_site_z_mean",
        "y_reg_next_admits"
    ]

    if state is not None:
        required_columns.append("state_territory")

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")

    # Filter by state
    if state is not None:
        data["state_territory"] = (
            data["state_territory"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        state_clean = str(state).strip().upper()

        data = data[
            data["state_territory"] == state_clean
        ].copy()

        if data.empty:
            raise ValueError(f"No rows found for state '{state}'.")

    # Convert columns to the correct data types
    data["week_end"] = pd.to_datetime(
        data["week_end"],
        errors="coerce"
    )

    data["conc_site_z_mean"] = pd.to_numeric(
        data["conc_site_z_mean"],
        errors="coerce"
    )

    data["y_reg_next_admits"] = pd.to_numeric(
        data["y_reg_next_admits"],
        errors="coerce"
    )

    # Remove unusable rows
    data = data.dropna(
        subset=[
            "week_end",
            "conc_site_z_mean",
            "y_reg_next_admits"
        ]
    )

    # Filter by year
    if year is not None:
        data = data[data["week_end"].dt.year == year].copy()

    if data.empty:
        raise ValueError(
            "No valid data remains after applying the filters."
        )

    data = data.sort_values("week_end")

    # Scale each variable separately to the range 0–1
    data["zscore_scaled"] = MinMaxScaler().fit_transform(
        data[["conc_site_z_mean"]]
    )

    data["next_admits_scaled"] = MinMaxScaler().fit_transform(
        data[["y_reg_next_admits"]]
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        data["week_end"],
        data["zscore_scaled"],
        linewidth=2,
        label="Wastewater Z-score (conc_site_z_mean)"
    )

    ax.plot(
        data["week_end"],
        data["next_admits_scaled"],
        linewidth=2,
        label="Next-Week Admissions (y_reg_next_admits)"
    )

    # Use monthly labels for one year and yearly labels otherwise
    if year is not None:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        x_label = str(year)
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        x_label = "Year"

    title = "Wastewater Z-score vs Next-Week Hospital Admissions"

    if state is not None:
        title += f" — {state}"

    if year is not None:
        title += f" ({year})"

    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalized Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# plot_trends(rolling_data, "tx") 
# plot_trends_2023(rolling_data, "ca")

# plot_zscore_vs_next_admits(rolling_data, state="ca", year=2023)