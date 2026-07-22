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


def plot_correlation_heatmap(df):
    """
    Plot a Pearson correlation matrix for selected wastewater and
    hospital-admission variables.

    Parameters:
        df (pd.DataFrame): Input dataframe.
    """

    columns = [
        "log10_conc_mean",
        "log10_conc_median",
        "conc_site_z_mean",
        "log10_conc_lag1",
        "log10_conc_lag2",
        "log10_conc_lag3",
        "conc_delta_1w",
        "conc_roll3",
        "admits_this_week",
        "admits_per100k",
        "y_reg_next_admits"
    ]

    # Check that all required columns exist
    missing_columns = [
        column for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")

    data = df[columns].copy()

    # Convert each column to numeric
    for column in columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Calculate the Pearson correlation matrix
    correlation_matrix = data.corr(method="pearson")

    plt.figure(figsize=(13, 10))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation"}
    )

    plt.title(
        "Correlation Matrix: Wastewater Features and Hospital Admissions",
        pad=15
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return correlation_matrix

def plot_wastewater_relationships(df):
    """
    Plot scatter plots comparing wastewater metrics with
    next-week hospital admissions.

    Includes:
        - log10_conc_mean vs y_reg_next_admits
        - conc_site_z_mean vs y_reg_next_admits
        - conc_roll3 vs y_reg_next_admits
        - admits_this_week vs y_reg_next_admits
    """

    predictors = [
        "log10_conc_mean",
        "conc_site_z_mean",
        "conc_roll3",
        "admits_this_week"
    ]

    target = "y_reg_next_admits"

    # Check columns
    required = predictors + [target]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    data = df[required].copy()

    # Convert to numeric
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, predictor in zip(axes, predictors):

        subset = data[[predictor, target]].dropna()

        sns.regplot(
            data=subset,
            x=predictor,
            y=target,
            ax=ax,
            scatter_kws={"alpha": 0.5, "s": 20},
            line_kws={"linewidth": 2}
        )

        r = subset[predictor].corr(subset[target])

        ax.set_title(f"{predictor}\n$r$ = {r:.2f}")
        ax.set_xlabel(predictor)
        ax.set_ylabel(target)

    plt.suptitle(
        "Wastewater Metrics vs Next-Week Hospital Admissions",
        fontsize=16,
        y=1.02
    )

    plt.tight_layout()
    plt.show()

def plot_lag_relationships(df):
    """
    Compare wastewater concentration lags with next-week admissions.

    Plots:
        - log10_conc_lag1 vs y_reg_next_admits
        - log10_conc_lag2 vs y_reg_next_admits
        - log10_conc_lag3 vs y_reg_next_admits
    """

    lag_columns = [
        "log10_conc_lag1",
        "log10_conc_lag2",
        "log10_conc_lag3"
    ]

    target = "y_reg_next_admits"
    required_columns = lag_columns + [target]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")

    data = df[required_columns].copy()

    # Convert values to numeric
    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        sharey=True
    )

    correlations = {}

    for ax, lag_column in zip(axes, lag_columns):
        subset = data[[lag_column, target]].dropna()

        if len(subset) < 2:
            ax.text(
                0.5,
                0.5,
                "Not enough valid data",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
            ax.set_title(lag_column)
            correlations[lag_column] = float("nan")
            continue

        correlation = subset[lag_column].corr(subset[target])
        correlations[lag_column] = correlation

        sns.regplot(
            data=subset,
            x=lag_column,
            y=target,
            ax=ax,
            scatter_kws={
                "alpha": 0.4,
                "s": 20
            },
            line_kws={
                "linewidth": 2
            }
        )

        ax.set_title(
            f"{lag_column} vs Next-Week Admissions\n"
            f"Pearson r = {correlation:.2f}"
        )
        ax.set_xlabel(lag_column)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("y_reg_next_admits")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    plt.suptitle(
        "Comparison of Wastewater Lag Features",
        fontsize=16
    )
    plt.tight_layout()
    plt.show()

    # Print correlations from strongest to weakest
    correlation_series = pd.Series(
        correlations,
        name="correlation_with_target"
    ).sort_values(
        key=lambda values: values.abs(),
        ascending=False
    )

    print("\nLag correlations with y_reg_next_admits:")
    print(correlation_series)

    return correlation_series

def plot_velocity_relationship(df):
    """
    Plot the relationship between the one-week change in wastewater
    concentration and next-week hospital admissions.

    Plots:
        conc_delta_1w vs y_reg_next_admits
    """

    predictor = "conc_delta_1w"
    target = "y_reg_next_admits"

    missing_columns = [
        column for column in [predictor, target]
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")

    data = df[[predictor, target]].copy()

    # Convert both columns to numeric
    data[predictor] = pd.to_numeric(
        data[predictor],
        errors="coerce"
    )

    data[target] = pd.to_numeric(
        data[target],
        errors="coerce"
    )

    # Remove rows with missing values
    data = data.dropna()

    if len(data) < 2:
        raise ValueError("Not enough valid data points to create the plot.")

    correlation = data[predictor].corr(data[target])

    plt.figure(figsize=(9, 6))

    sns.regplot(
        data=data,
        x=predictor,
        y=target,
        scatter_kws={
            "alpha": 0.4,
            "s": 25
        },
        line_kws={
            "linewidth": 2
        }
    )

    # Mark where wastewater concentration is unchanged
    plt.axvline(
        x=0,
        linestyle="--",
        linewidth=1,
        label="No weekly concentration change"
    )

    plt.xlabel("One-Week Change in Wastewater Concentration")
    plt.ylabel("Next-Week Hospital Admissions")
    plt.title(
        "Wastewater Velocity vs Next-Week Hospital Admissions\n"
        f"Pearson r = {correlation:.2f}"
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return correlation

def plot_rolling_average_relationship(df):
    """
    Plot the relationship between the 3-week rolling wastewater
    concentration average and future hospital admissions.
    """

    predictor = "conc_roll3"
    target = "y_reg_next_admits"

    missing_columns = [
        column for column in [predictor, target]
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")

    data = df[[predictor, target]].copy()

    # Convert columns to numeric and remove invalid rows
    data[predictor] = pd.to_numeric(
        data[predictor],
        errors="coerce"
    )
    data[target] = pd.to_numeric(
        data[target],
        errors="coerce"
    )
    data = data.dropna()

    if len(data) < 2:
        raise ValueError(
            "Not enough valid data points to create the plot."
        )

    correlation = data[predictor].corr(data[target])

    plt.figure(figsize=(9, 6))

    sns.regplot(
        data=data,
        x=predictor,
        y=target,
        scatter_kws={
            "alpha": 0.4,
            "s": 25
        },
        line_kws={
            "linewidth": 2
        }
    )

    plt.xlabel("3-Week Rolling Wastewater Concentration")
    plt.ylabel("Future Hospital Admissions")
    plt.title(
        "Rolling Wastewater Average vs Future Admissions\n"
        f"Pearson r = {correlation:.2f}"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return correlation

# correlation_matrix = plot_correlation_heatmap(rolling_data)
# print(
#     correlation_matrix["y_reg_next_admits"]
#     .sort_values(ascending=False)
# )

# plot_wastewater_relationships(rolling_data)

# lag_correlations = plot_lag_relationships(rolling_data)
# velocity_correlation = plot_velocity_relationship(rolling_data)
# rolling_correlation = plot_rolling_average_relationship(rolling_data)