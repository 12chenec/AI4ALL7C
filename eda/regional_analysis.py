# This code is for creating data visualizations / data exploration
# specifically for regional patterns, focused on state/territory and 
# county. County is defined by County FIP codes. 

# Import Packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the wastewater dataset
wastewater_dataset = pd.read_csv(
    "/Users/christalchen/Desktop/AI4ALL/CDC_Wastewater_Data_for_SARS-CoV-2.csv",
)
# Load the counties dataset
counties_dataset = pd.read_csv(
    "/Users/christalchen/Desktop/AI4ALL/uscounties.csv",
)
# Load the urbanization dataset
urban_dataset = pd.read_csv(
    "/Users/christalchen/Desktop/AI4ALL/Ruralurbancontinuumcodes2023.csv",
    encoding='latin1',
)
# Load the merged datasets
rolling_data = pd.read_csv("model_ready_weekly.csv")
era_data = pd.read_csv("model_ready_era2022.csv")

# Count the number of records in each state/territory.
def plot_samples_by_territory(df):
    sample_counts = df.groupby('state_territory').size().reset_index(name='sample_count')
    plt.figure(figsize=(8, 4))
    sns.barplot(x='state_territory', y='sample_count', data=sample_counts)
    plt.title('Number of Samples by State/Territory')
    plt.ylabel('Number of Samples')
    plt.show()


# Find the county FIPS column in a dataframe.
def _get_county_fips_column(df):
    for column_name in ('county_fips', 'FIPS'):
        if column_name in df.columns:
            return column_name
    raise KeyError('No county FIPS column found')


# Find the state column in a dataframe.
def _get_state_column(df):
    for column_name in ('state_territory', 'state_id', 'State'):
        if column_name in df.columns:
            return column_name
    raise KeyError('No state column found')


# Extract unique county FIPS values from a dataframe.
def _extract_county_fips(df):
    fips_column = _get_county_fips_column(df)
    county_fips_values = set()
    for raw_value in df[fips_column].dropna().astype(str):
        for county_code in raw_value.split(','):
            county_fips_values.add(county_code.strip().zfill(5))
    return county_fips_values


# Print the county coverage percentage against a county reference file.
def print_county_coverage_percentage(wastewater_df, counties_df):
    wastewater_counties = _extract_county_fips(wastewater_df)
    all_counties = set(counties_df['county_fips'].dropna().astype(str).str.zfill(5))
    represented_counties = wastewater_counties & all_counties

    coverage_percentage = (len(represented_counties) / len(all_counties)) * 100
    print(f'{len(represented_counties)} of {len(all_counties)} counties represented ({coverage_percentage:.2f}%)')


# Count the number of unique wastewater sites in each state/territory.
def plot_sites_by_territory(df):
    site_counts = df.groupby('state_territory')['site'].nunique().reset_index(name='site_count')
    plt.figure(figsize=(8, 4))
    sns.barplot(x='state_territory', y='site_count', data=site_counts)
    plt.title('Number of Sites by State/Territory')
    plt.ylabel('Number of Sites')
    plt.show()


# Compare county coverage between wastewater and county reference data by territory.
def plot_unrepresented_counties_by_territory(wastewater_df, counties_df):
    wastewater_by_state = {}
    for state_territory, group in wastewater_df.groupby('state_territory'):
        county_fips_values = _extract_county_fips(group)
        wastewater_by_state[state_territory.upper()] = county_fips_values

    county_totals = counties_df.groupby('state_id')['county_fips'].nunique()

    plot_rows = []
    for state_id, total_counties in county_totals.items():
        represented_counties = wastewater_by_state.get(state_id, set())
        unrepresented_counties = total_counties - len(represented_counties & set(
            counties_df.loc[counties_df['state_id'] == state_id, 'county_fips'].astype(str).str.zfill(5)
        ))
        percentage_unrepresented = (unrepresented_counties / total_counties) * 100
        plot_rows.append({
            'state_territory': state_id.lower(),
            'unrepresented_percentage': percentage_unrepresented,
        })

    plot_data = pd.DataFrame(plot_rows).sort_values('state_territory')
    plt.figure(figsize=(12, 5))
    sns.barplot(x='state_territory', y='unrepresented_percentage', data=plot_data)
    plt.title('Percentage of Unrepresented Counties by State/Territory')
    plt.ylabel('Unrepresented Counties (%)')
    plt.xlabel('State/Territory')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# Print how many county FIPS values appear only in one of two datasets.
def print_county_difference_counts(first_df, second_df):
    first_counties = _extract_county_fips(first_df)
    second_counties = _extract_county_fips(second_df)

    first_only_count = len(first_counties - second_counties)
    second_only_count = len(second_counties - first_counties)

    print(f'Counties in the first dataset but not the second: {first_only_count}')
    print(f'Counties in the second dataset but not the first: {second_only_count}')


# Compare average RUCC values for counties represented and not represented in wastewater.
def print_average_rucc_for_represented_and_unrepresented_counties(wastewater_df, urban_df):
    wastewater_states = set(wastewater_df['state_territory'].dropna().astype(str).str.upper())
    wastewater_counties = _extract_county_fips(wastewater_df)

    state_column = _get_state_column(urban_df)
    rucc_data = urban_df[urban_df['Attribute'].astype(str).eq('RUCC_2023')].copy()
    rucc_data['FIPS'] = rucc_data['FIPS'].astype(str).str.zfill(5)
    rucc_data['Value'] = pd.to_numeric(rucc_data['Value'], errors='coerce')
    rucc_data = rucc_data[rucc_data[state_column].astype(str).str.upper().isin(wastewater_states)]

    represented_counties = rucc_data[rucc_data['FIPS'].isin(wastewater_counties)]
    unrepresented_counties = rucc_data[~rucc_data['FIPS'].isin(wastewater_counties)]

    represented_average = represented_counties['Value'].mean()
    unrepresented_average = unrepresented_counties['Value'].mean()

    print(f'Average RUCC for represented counties: {represented_average:.2f}')
    print(f'Average RUCC for unrepresented counties: {unrepresented_average:.2f}')


# Build a ranked list of PCR methods by overall frequency.
def _prepare_pcr_method_counts(wastewater_df):
    method_counts = (
        wastewater_df['pcr_type']
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
        .value_counts()
        .sort_values(ascending=False)
    )

    if len(method_counts) > 3:
        other_count = method_counts.iloc[-3:].sum()
        method_counts = method_counts.iloc[:-3]
        method_counts['other'] = other_count

    return method_counts


# Assign consistent colors to PCR methods.
def _get_pcr_method_colors(method_names):
    palette = [
        '#4C78A8',
        '#F58518',
        '#54A24B',
        '#E45756',
        '#72B7B2',
        '#EECA3B',
        '#B279A2',
        '#FF9DA6',
        '#9D755D',
        '#BAB0AC',
    ]
    return {method_name: palette[index % len(palette)] for index, method_name in enumerate(method_names)}


# Plot the PCR method mix by state/territory using consistent method order and colors.
def plot_pcr_method_by_state(wastewater_df):
    method_data = wastewater_df[['state_territory', 'pcr_type']].dropna().copy()
    method_data['pcr_type'] = method_data['pcr_type'].astype(str).str.lower().str.strip()

    method_series = _prepare_pcr_method_counts(wastewater_df)
    method_order = list(method_series.index)
    method_colors = _get_pcr_method_colors(method_order)
    top_methods = method_order[:-1] if 'other' in method_order else method_order
    method_data['pcr_type'] = method_data['pcr_type'].where(method_data['pcr_type'].isin(top_methods), 'other')

    method_counts = (
        method_data.groupby(['state_territory', 'pcr_type'])
        .size()
        .reset_index(name='count')
    )
    method_totals = method_counts.groupby('state_territory')['count'].transform('sum')
    method_counts['share'] = (method_counts['count'] / method_totals) * 100

    state_ddpcr_share = (
        method_counts[method_counts['pcr_type'] == 'ddpcr']
        .set_index('state_territory')['share']
        .sort_values(ascending=False)
    )
    state_order = state_ddpcr_share.index.tolist()

    plot_data = method_counts.pivot(index='state_territory', columns='pcr_type', values='share').fillna(0)
    plot_data = plot_data.reindex(columns=method_order, fill_value=0)
    plot_data = plot_data.reindex(state_order)

    plt.figure(figsize=(16, 7))
    bottom = pd.Series([0] * len(plot_data), index=plot_data.index)
    for method_name in method_order:
        if method_name in plot_data.columns:
            plt.bar(
                plot_data.index,
                plot_data[method_name],
                bottom=bottom,
                label=method_name,
                color=method_colors[method_name],
            )
            bottom = bottom + plot_data[method_name]

    plt.title('PCR Method Mix by State/Territory')
    plt.ylabel('Share of Rows (%)')
    plt.xlabel('State/Territory')
    plt.xticks(rotation=90)
    plt.legend(title='PCR Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# Plot overall PCR method popularity in the wastewater dataset.
def plot_pcr_method_popularity(wastewater_df):
    method_counts = _prepare_pcr_method_counts(wastewater_df)
    method_colors = _get_pcr_method_colors(method_counts.index)

    plt.figure(figsize=(10, 10))
    plt.pie(
        method_counts.values,
        labels=method_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=[method_colors[method_name] for method_name in method_counts.index],
    )
    plt.axis('equal')
    plt.title('PCR Method Popularity in the Wastewater Dataset')
    plt.tight_layout()
    plt.show()


# Compare PCR target concentration across concentration methods.
def plot_concentration_method_effect_on_measurements(wastewater_df):
    plot_data = wastewater_df[['concentration_method', 'pcr_target_avg_conc']].copy()
    plot_data['concentration_method'] = plot_data['concentration_method'].astype(str).str.lower().str.strip()
    plot_data['pcr_target_avg_conc'] = pd.to_numeric(plot_data['pcr_target_avg_conc'], errors='coerce')
    plot_data = plot_data.dropna(subset=['concentration_method', 'pcr_target_avg_conc'])

    method_order = (
        plot_data.groupby('concentration_method')['pcr_target_avg_conc']
        .median()
        .sort_values()
        .index
    )

    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=plot_data,
        x='pcr_target_avg_conc',
        y='concentration_method',
        order=method_order,
        showfliers=False,
    )
    plt.xscale('log')
    plt.title('PCR Target Concentration by Concentration Method')
    plt.xlabel('PCR Target Average Concentration (log scale)')
    plt.ylabel('Concentration Method')
    plt.tight_layout()
    plt.show()


# Plot average PCR target concentration by state/territory.
def plot_average_pcr_target_conc_by_territory(df):
    plot_data = df.copy()
    plot_data['pcr_target_avg_conc'] = pd.to_numeric(plot_data['pcr_target_avg_conc'], errors='coerce')
    average_conc = plot_data.groupby('state_territory')['pcr_target_avg_conc'].mean().reset_index()

    plt.figure(figsize=(12, 5))
    sns.barplot(x='state_territory', y='pcr_target_avg_conc', data=average_conc)
    plt.title('Average PCR Target Concentration by State/Territory')
    plt.ylabel('Average PCR Target Concentration')
    plt.xlabel('State/Territory')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# Plot average flow rate by state/territory.
def plot_average_flow_rate_by_territory(df):
    plot_data = df.copy()
    plot_data['flow_rate'] = pd.to_numeric(plot_data['flow_rate'], errors='coerce')
    average_flow_rate = plot_data.groupby('state_territory')['flow_rate'].mean().reset_index()

    plt.figure(figsize=(12, 5))
    sns.barplot(x='state_territory', y='flow_rate', data=average_flow_rate)
    plt.title('Average Flow Rate by State/Territory')
    plt.ylabel('Average Flow Rate')
    plt.xlabel('State/Territory')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# Summarize the wastewater features that vary most across territories.
def summarize_features_varying_most_by_territory(wastewater_df, top_n=10):
    territory_column = 'state_territory'
    numeric_columns = wastewater_df.select_dtypes(include='number').columns.tolist()
    excluded_columns = {'site'}
    numeric_columns = [column_name for column_name in numeric_columns if column_name not in excluded_columns]

    variation_rows = []
    for feature_name in numeric_columns:
        feature_data = wastewater_df[[territory_column, feature_name]].copy()
        feature_data[feature_name] = pd.to_numeric(feature_data[feature_name], errors='coerce')
        territory_means = feature_data.groupby(territory_column)[feature_name].mean().dropna()

        if len(territory_means) < 2:
            continue

        variation_rows.append({
            'feature': feature_name,
            'territory_mean_std': territory_means.std(),
            'territory_mean_range': territory_means.max() - territory_means.min(),
        })

    variation_data = pd.DataFrame(variation_rows)
    if variation_data.empty:
        print('No numeric features with enough territory coverage were found.')
        return

    variation_data = variation_data.sort_values('territory_mean_std', ascending=False).head(top_n)

    print('Features that vary most across territories:')
    for rank, row in enumerate(variation_data.itertuples(index=False), start=1):
        print(
            f'{rank}. {row.feature}: std={row.territory_mean_std:.4g}, '
            f'range={row.territory_mean_range:.4g}'
        )


# plot_samples_by_territory(era_data)
# plot_sites_by_territory(wastewater_dataset)
# print_county_coverage_percentage(wastewater_dataset, counties_dataset)
# plot_unrepresented_counties_by_territory(wastewater_dataset, counties_dataset)
# print_county_difference_counts(wastewater_dataset, counties_dataset)
# print_average_rucc_for_represented_and_unrepresented_counties(wastewater_dataset, urban_dataset)
# plot_pcr_method_by_state(wastewater_dataset)
# plot_pcr_method_popularity(wastewater_dataset)
# plot_concentration_method_effect_on_measurements(wastewater_dataset)
# plot_average_pcr_target_conc_by_territory(wastewater_dataset)
# plot_average_flow_rate_by_territory(wastewater_dataset)
# summarize_features_varying_most_by_territory(wastewater_dataset)