import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from contamination_model import config


def plot_configuration(x: float = 11.7, y: float = 8.27) -> None:
    """
    Fixed configuration for seaborn plots.

    Parameters
    ----------
    x : float, optional
        The plot's X-axis, by default 11.7
    y : float, optional
        The plot's X-axis, by default 8.27
    """
    a4_dims = (x, y)
    plt.subplots(figsize=a4_dims)


def plotting_categories(df: pd.DataFrame, column: str) -> None:
    """
    Distribution Plot for the selected column, 
    filtering by variable.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with a category column.
    column : str
        The category column for plotting.
    """
    categories = df[column].unique()

    for category in categories:
        series_to_plot = df[df[column] == category]['prob_V1_V2']
        sns.distplot(series_to_plot, hist=False, label=category)


def create_directories(directories_list: list) -> None:
    """
    Creates possible missing directories for model pipeline.
    Parameters
    ----------
    directories_list : List of path directories to be created.

    Returns
    -------

    """
    for directory in directories_list:
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass
