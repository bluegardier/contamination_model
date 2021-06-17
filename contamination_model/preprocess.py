import numpy as np
import pandas as pd
from typing import Tuple
from contamination_model import config, utils


def create_target_dataframe(
        df_target: pd.DataFrame, df_list: list
) -> pd.DataFrame:
    """
    Creates the model dataframe, merging with the previously
    created dataframes.

    Parameters
    ----------
    df_target : pd.DataFrame
        The dataframe containing the target variable.
    df_list : list
        List of datafgrames.

    Returns
    -------
    pd.DataFrame
        Dataframe ready for model.
    """

    target = df_target[~df_target["prob_V1_V2"].isnull()]

    for df in df_list:
        merge_key = df.columns[0]
        target = target.merge(df, on=merge_key, how="left")
        target.dropna(inplace=True)

    return target


def applying_suffix_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Renames the df columns, including a suffix.

    Parameters
    ----------
    df : pd.DataFrame
        The original Dataframe

    Returns
    -------
    pd.DataFrame
        Datafrae with renamed columns.
    """

    data = df.copy()

    data.columns = [str(col) + suffix for col in data.columns]
    data.rename(columns={"name{}".format(suffix): suffix[1:]}, inplace=True)

    return data


def split_for_train_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe for modelling.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The train/test and validation dataframes.
    """

    validation_data = df.sample(frac=0.75, random_state=16)
    train_test_data = df.drop(validation_data.index, axis=1)

    return train_test_data, validation_data


def rename_category(df: pd.DataFrame, sufix: str) -> pd.DataFrame:
    """
    Includes sufix for categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with categorical columns.
    sufix : str
        The desirable sufix.

    Returns
    -------
    pd.DataFrame
        Dataframe with renamed categories.
    """

    df = df.dropna()

    to_categorize = df.select_dtypes(include="object")

    for column in to_categorize:
        column_values = df[column].unique()

        for values in column_values:
            df[column] = df[column].replace(values, values + sufix)

    return df


def refactor_binary_missing_variables(
        df: pd.DataFrame, variable_list: list
) -> pd.DataFrame:
    """
    Renames variable's categories instead of using int numbers.
    One of the main purposes of this function is to have a 
    cleaner visual dataframe for plotting, modelling and other
    methods.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with categorical features.
    variable_list : list
        A list with categorical features as string.

    Returns
    -------
    pd.DataFrame
        The dataframe with renamed feature categories.
    """

    data = df.copy()

    for var in variable_list:
        data["{}_status".format(var)] = np.where(
            data[var] == 0, "nao_{}".format(var),
            np.where(
                data[var] == 1, var, "sem_info_{}".format(var)
            )
        )

    data.drop(variable_list, axis=1, inplace=True)

    return data


def refactor_counting_missing_variables(
        df: pd.DataFrame, variable_list: list, category_name: str
) -> pd.DataFrame:
    """
    Refactor counting variables to category.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with counting features.
    variable_list : list
        A list with counting features as string.
    category_name : str
        Name of the new category.

    Returns
    -------
    pd.DataFrame
        The dataframe with the selected variables refactored.
    """

    data = df.copy()

    for var in variable_list:
        data["{}_status".format(var)] = np.where(
            data[var] == 0, "sem_{}".format(category_name),
            np.where(
                data[var] > 0, category_name, df[var].mode()[0]
            )
        )

    data.drop(variable_list, axis=1, inplace=True)

    return data


def filling_missings(
        df: pd.DataFrame, variable_list: list, fill_method: str = "mode"
) -> pd.DataFrame:
    """
    Fill missing values for a number of categorical variables
    with their's mode.

    Parameters
    ----------
    fill_method : Fill the dataset with Mode. Uses median if else.
    df : pd.DataFrame
        Dataframe with categorical features.
    variable_list : list
        List of variables for missing imputing.
    fill_mode : str
        Fills missing values with mode by default. Uses Median otherwise.

    Returns
    -------
    pd.DataFrame
        The dataframe with missing values filled.
    """
    data = df.copy()

    for var in variable_list:
        if fill_method == "mode":
            data[var].fillna(data[var].mode()[0], inplace=True)
        else:
            data[var].fillna(data[var].median(), inplace=True)

    return data


def create_status_imc_variable(df: pd.DataFrame) -> np.ndarray:
    """
    Creates the "status_IMC" variable.

    Parameters
    ----------
    df : pd.DataFrame
        The Dataframe with "IMC" variable.

    Returns
    -------
    np.array
        A np.array with the categorized "IMC" feature.
    """

    imc_categories = np.where(
        df["IMC"] < 17,
        "muito_abaixo",
        np.where((df["IMC"] >= 17) &
                 (df["IMC"] < 18.5),
                 "abaixo",
                 np.where((df["IMC"] >= 18.5) &
                          (df["IMC"] < 25),
                          "adequado",
                          np.where((df["IMC"] >= 25) &
                                   (df["IMC"] < 30),
                                   "acima",
                                   np.where((df["IMC"] >= 30) &
                                            (df["IMC"] < 35),
                                            "obesidade_I",
                                            np.where((df["IMC"] >= 35) &
                                                     (df["IMC"] < 40),
                                                     "obesidade_II",
                                                     np.where((
                                                             df["IMC"] >= 40),
                                                         "obsidade_III",
                                                         "sem_info_imc"
                                                     )
                                                     )
                                            )
                                   )
                          )
                 )
    )
    return imc_categories


def create_faixa_etaria_variable(df: pd.DataFrame) -> np.ndarray:
    """
    Creates "faixa etaria" variable.

    Parameters
    ----------
    df : pd.DataFrame
        The Dataframe with "idade" variable.

    Returns
    -------
    np.ndarray
        A np.array with the categorized "idade" feature.
    """

    faixa_etaria = np.where(
        df["idade"] < 18, "menor_18",
        np.where((df["idade"] >= 18) &
                 (df["idade"] < 25),
                 "18_24_anos",
                 np.where((
                                  df["idade"] >= 25) &
                          (df["idade"] < 35),
                          "25_34_anos",
                          np.where((
                                           df["idade"] >= 35) &
                                   (df["idade"] < 45),
                                   "35_44_anos",
                                   np.where((
                                                    df["idade"] >= 45) &
                                            (df["idade"] < 55),
                                            "45_54_anos",
                                            np.where((
                                                             df["idade"] >= 55) &
                                                     (df["idade"] < 65),
                                                     "55_64_anos",
                                                     np.where((
                                                             df["idade"] >= 65),
                                                         "maior_65",
                                                         "none")
                                                     )
                                            )
                                   )
                          )
                 )
    )

    return faixa_etaria


def preprocess_predict_data(
        df_target: pd.DataFrame,
        df_v1: pd.DataFrame,
        df_v2: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates the dataframe for target prediction.

    Parameters
    ----------
    df_target : pd.DataFrame
        The dataframe with the target variable.
    df_v1 : pd.DataFrame
        Dataframe containing info about the V1 person.
    df_v2 : pd.DataFrame
        Dataframe containing info about the V2 person.

    Returns
    -------
    pd.DataFrame
        Dataframe for prediction.
    """
    predict_df = df_target[df_target["prob_V1_V2"].isnull()]
    predict_df = predict_df.merge(df_v1, on="V1", how="left")
    predict_df = predict_df.merge(df_v2, on="V2", how="left")

    return predict_df


def preprocess_data(
        df: pd.DataFrame, df_target: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compiles all methods to create the model's dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing individual data.
    df_target : pd.DataFrame
        DataFrame containing the contamination probability.

    Returns
    -------
    pd.DataFrame
    """
    utils.create_directories([config.models_path, config.processed_data_path])

    df = refactor_counting_missing_variables(
        df, ["qt_filhos"], "filhos")

    to_fillna = df.select_dtypes(include="object").columns.to_list()

    df = filling_missings(df, config.binary_variables)
    df = filling_missings(df, to_fillna)
    df = filling_missings(
        df, config.median_fill_variables, fill_method="median")
    df = refactor_binary_missing_variables(
        df, config.binary_variables)
    df["faixa_etaria"] = create_faixa_etaria_variable(df)
    df["status_IMC"] = create_status_imc_variable(df)

    df01 = rename_category(df, "__V1")
    df02 = rename_category(df, "__V2")

    df01 = applying_suffix_columns(df01, "_V1")
    df02 = applying_suffix_columns(df02, "_V2")

    df_list = [df01, df02]

    final_df = create_target_dataframe(df_target, df_list)

    return final_df, df01, df02

