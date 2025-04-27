import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


PLOTS_DIR = "../Plots/"
HIST_DIR = "../Hist/"

# Set font sizes for plots
def set_font_sizes():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18
    })


def clear_plot():
    plt.close()
    plt.clf()
    plt.figure(figsize=(8, 6))


# Create histograms for all columns
def all_hist(df: pd.DataFrame):
    column_names = {
        "tickets": "Wysokość mandatu w euro",
        "cars per 1k": "Samochody na 1000 mieszkańców",
        "pop": "Populacja",
        "land area": "Powierzchnia kraju",
        "death ratio": "Śmierci na milion mieszkańców",
        "percent old cars": "Procent samochodów starszych niż 10 lat",
        "region": "Region"
    }

    for column in df.columns:
        clear_plot()
        sns.histplot(data=df, x=column, bins=15)
        plt.xlabel(column_names[column])
        plt.ylabel("Częstość")
        plt.savefig(f"{HIST_DIR}{column}_hist.png")


# Create a plot with one linear and three polynomial regression models
def lin_glm_plot(x, y):
    clear_plot()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_axis = np.linspace(start=x.min(), stop=x.max(), num=300)

    lin_plot(x_axis, x_test, x_train, y_test, y_train)
    glm_plot(x_axis, x_test, x_train, y_test, y_train, degree=2, color="orange")
    glm_plot(x_axis, x_test, x_train, y_test, y_train, degree=3, color="green")
    glm_plot(x_axis, x_test, x_train, y_test, y_train, degree=4, color="blue")

    plt.scatter(x_train, y_train, label="Dane treningowe", alpha=0.7)
    plt.scatter(x_test, y_test, edgecolor="black", facecolor="none", label="Dane testujące")

    plt.legend()


# Create a plot with one polynomial regression model
def glm_plot(x_axis, x_test, x_train, y_test, y_train, degree=3, color="green"):
    model_glm = LinearRegression()
    gen_features = PolynomialFeatures(degree=degree, include_bias=True, interaction_only=False)
    model_glm.fit(gen_features.fit_transform(x_train.reshape(-1, 1)), y_train)

    y_glm_pred = model_glm.predict(gen_features.fit_transform(x_axis.reshape(-1, 1)))
    mse_glm = mean_squared_error(y_test, model_glm.predict(gen_features.fit_transform(x_test.reshape(-1, 1))))

    print(f"MSE for GLM{degree}: {mse_glm}")
    plt.plot(x_axis, y_glm_pred, label=f"Model GLM{degree}", color=color)


# Create a plot with one linear regression model
def lin_plot(x_axis, x_test, x_train, y_test, y_train, color="red"):
    model_lin = LinearRegression()
    model_lin.fit(x_train.reshape(-1, 1), y_train)

    y_lin_pred = model_lin.predict(x_axis.reshape(-1, 1))
    mse_lin = mean_squared_error(y_test, model_lin.predict(x_test.reshape(-1, 1)))

    print(f"MSE for linear model: {mse_lin}")
    plt.plot(x_axis, y_lin_pred, color=color, label="Model liniowy")


def ticket_plot(df: pd.DataFrame):
    x = df["tickets"].values
    y = df["death ratio"].values

    # delete 1 outlier
    max_index = np.argmax(x)
    x = np.delete(x, max_index)
    y = np.delete(y, max_index)

    print("Ticket plot")
    lin_glm_plot(x, y)

    plt.xlabel("Wysokość mandatu w euro")
    plt.ylabel("Śmierci na milion mieszkańców")
    plt.savefig(f"{PLOTS_DIR}ticket_plot.png")


def car_age_plot(df: pd.DataFrame):
    x = df["percent old cars"].values
    y = df["death ratio"].values

    print("Car age plot")
    lin_glm_plot(x, y)

    plt.xlabel("Procent samochodów starszych niż 10 lat")
    plt.ylabel("Śmierci na milion mieszkańców")
    plt.savefig(f"{PLOTS_DIR}car_age_plot.png")


def cars_per_1k_plot(df: pd.DataFrame):
    x = df["cars per 1k"].values
    y = df["death ratio"].values

    # delete 2 outliers
    for i in range(2):
        min_index = np.argmin(x)
        x = np.delete(x, min_index)
        y = np.delete(y, min_index)

    print("Cars per 1k plot")
    lin_glm_plot(x, y)

    plt.xlabel("Samochody na 1000 mieszkańców")
    plt.ylabel("Śmierci na milion mieszkańców")
    plt.savefig(f"{PLOTS_DIR}cars_per_1k_plot.png")


# Create a scatter plot with death ratio on the y-axis and region on the x-axis, with different colors for each region
def region_plot(df: pd.DataFrame):
    clear_plot()
    regions = sorted(df["region"].unique())
    colors = plt.cm.get_cmap("Set3", len(regions))

    for i, region in enumerate(regions):
        subset = df[df["region"] == region]
        plt.scatter(subset["region"], subset["death ratio"], color=colors(i), label=region)

    plt.xlabel("Region")
    plt.ylabel("Śmierci na milion mieszkańców")
    plt.savefig(f"{PLOTS_DIR}region_plot.png")


# Create a bar plot with the mean death ratio for each region, with different colors for each region
def region_bar_plot(df: pd.DataFrame):
    mean_death_ratio = df.groupby("region")["death ratio"].mean()
    clear_plot()
    plt.bar(mean_death_ratio.index, mean_death_ratio, color=sns.color_palette("Set3"))
    plt.xlabel("Region")
    plt.ylabel("Średnia śmierci na milion mieszkańców")

    plt.savefig(f"{PLOTS_DIR}region_bar_plot.png")


# Create a map with the death ratio for each country
def death_ratio_map(df: pd.DataFrame):
    clear_plot()
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    europe = world[(world["continent"] == "Europe") & (world["name"] != "Russia")]
    europe = europe.merge(df[["death ratio"]], left_on="name", right_index=True, how="left")

    plt.xlim(-30, 40)
    plt.ylim(35, 75)
    plt.xlabel("Długość geograficzna")
    plt.ylabel("Szerokość geograficzna")

    plt.title("Śmierci na milion mieszkańców")
    ax = plt.gca()
    europe.boundary.plot(ax=ax)
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    europe.plot(column="death ratio", ax=ax, legend=True, cmap=cmap, missing_kwds={"color": "lightgrey"})
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}death_ratio_countries_map.png")


# Create a map with the region for each country
def region_map(df: pd.DataFrame):
    clear_plot()
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    europe = world[(world["continent"] == "Europe") & (world["name"] != "Russia")]
    europe = europe.merge(df[["region"]], left_on="name", right_index=True, how="left")

    plt.xlim(-30, 40)
    plt.ylim(35, 75)
    plt.xlabel("Długość geograficzna")
    plt.ylabel("Szerokość geograficzna")

    ax = plt.gca()
    plt.title("Regiony")
    europe.boundary.plot(ax=ax)
    cmp = sns.color_palette("Set3", as_cmap=True)
    europe.plot(column="region", ax=ax, legend=True, cmap=cmp, missing_kwds={"color": "lightgrey",
                                                                             "label": "Brak danych"})
    plt.savefig(f"{PLOTS_DIR}region_map.png")


# Create a heatmap with the correlation matrix of the death ratio, tickets, cars per 1k and percent old cars columns
def corr_heatmap(df: pd.DataFrame):
    clear_plot()
    ax = plt.gca()
    cmp = sns.color_palette("coolwarm", as_cmap=True)
    sns.heatmap(df[["death ratio", "tickets", "cars per 1k", "percent old cars"]].corr(), annot=True, ax=ax, cmap=cmp)
    plt.savefig(f"{PLOTS_DIR}corr_heatmap.png")
