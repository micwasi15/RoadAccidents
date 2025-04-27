import pandas as pd


def get_dataframe():
    filepath = "../Datasets/Data.xlsx"
    df = pd.read_excel(filepath, index_col=0)
    df.rename(columns={"Tickets - Euro": "tickets",
                       "Passenger cars - per thousand inhabitants": "cars per 1k",
                       "Population": "pop",
                       "Deaths in road accidents": "deaths",
                       "Passenger cars older than 10 years": "old cars",
                       "Passenger cars": "cars",
                       "Region": "region"},
              inplace=True)

    # Attach land area data
    df = attach_land_area(df)

    # Create new columns
    df["death ratio"] = df["deaths"] / df["pop"] * 1000000
    df["percent old cars"] = df["old cars"] / df["cars"] * 100

    # Fill missing values in percent old cars and tickets columns with the means of the region
    regions_mean = df.groupby("region")["percent old cars"].transform("mean")
    df["percent old cars"].fillna(regions_mean, inplace=True)
    regions_mean = df.groupby("region")["tickets"].transform("mean")
    df["tickets"].fillna(regions_mean, inplace=True)

    # Replace region codes with region names
    df["region"] = df["region"].replace(
        {"N": "Skandynawia", "S": "Południe", "E": "Wschód", "W": "Zachód", "B": "Bałkany"})

    # Drop rows with missing values in the death ratio column
    df.dropna(inplace=True, subset=["death ratio"])

    # Drop old cars, cars and deaths columns
    df.drop(columns=["old cars",
                     "cars",
                     "deaths"],
            inplace=True)

    return df


def attach_land_area(df: pd.DataFrame):
    # Load land area data
    filepath = "../Datasets/land_area.csv"
    df_land_area = pd.read_csv(filepath, index_col=0, skiprows=4)

    # Choose only the 2019 data
    df_land_area = df_land_area[["2019"]]
    # Rename the column to land area and change the index name for Slovakia
    df_land_area.rename(columns={"2019": "land area"}, inplace=True, index={"Slovak Republic": "Slovakia"})
    # Add Kosovo land area which is missing in the file
    df_land_area.loc["Kosovo", "land area"] = 10887

    # Merge the dataframes
    res = pd.concat([df, df_land_area], axis=1, join="inner")

    return res
