# Import Packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the merged datasets
rolling_data = pd.read_csv("model_ready_weekly.csv")
era_data = pd.read_csv("model_ready_era2022.csv")

def print_column_names(df):
    """
    Takes a pandas DataFrame and prints all of its column names.
    """
    for column in df.columns:
        print(column)

# print_column_names(rolling_data)
print_column_names(era_data)