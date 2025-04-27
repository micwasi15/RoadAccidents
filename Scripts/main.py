import os.path

import pandas as pd

from Scripts.LoadingData import get_dataframe
from Scripts.Plots import death_ratio_map, region_map, corr_heatmap, car_age_plot, region_bar_plot, \
    all_hist, set_font_sizes, cars_per_1k_plot, ticket_plot, region_plot, PLOTS_DIR, HIST_DIR


# Save statistics (min, max, median, mean) for each numerical column to a textfile
def save_stats(df: pd.DataFrame):
    stats = []

    for column in df.select_dtypes(include=["number"]).columns:
        min_val = df[column].min()
        max_val = df[column].max()
        median_val = df[column].median()
        mean_val = df[column].mean()

        stats.append(f"{column}:\tMin: {min_val}\tMax: {max_val}\tMedian: {median_val}\tMean: {mean_val}\t")

    with open("stats.txt", "w") as f:
        for stat in stats:
            f.write(stat + '\n')


def main():
    # Load and clean data
    df = get_dataframe()

    # Save cleaned data to an Excel file and save statistics
    df.to_excel("../Datasets/cleaned_data.xlsx")
    save_stats(df)

    # Print some basic info about the data
    print(df[["death ratio", "region"]].sort_values(by="death ratio", ascending=False))
    print(df.info())

    # Set font sizes for plots
    set_font_sizes()

    # Create a directory for plots if it doesn't exist
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # Create a directory for histograms if it doesn't exist
    if not os.path.exists(HIST_DIR):
        os.makedirs(HIST_DIR)

    # Create histograms for all columns
    all_hist(df)

    # Create plots and save them as png files in the plots folder
    ticket_plot(df)
    car_age_plot(df)
    region_plot(df)
    region_bar_plot(df)
    cars_per_1k_plot(df)

    death_ratio_map(df)
    region_map(df)

    corr_heatmap(df)


if __name__ == "__main__":
    main()
