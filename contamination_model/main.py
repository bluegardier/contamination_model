import pandas as pd
import fire
import pickle
from contamination_model import config, preprocess, modelling


def features(df_path: str = config.DF_PATH, df_target_path: str = config.DF_TARGET_PATH) -> None:
    """
    Generates the features to create the train and test dataframes for
    model stage.

    Parameters
    ----------
    df_path : str, optional
        Path to data for individual analysis, by default df_path
    df_target_path : str, optional
        Path to data for connection analysis, by default df_target_path

    """
    df = pd.read_csv(df_path, sep=";")
    df_target = pd.read_csv(df_target_path, sep=";")

    print("Creating Train Dataframe.")
    df_train, df_v1, df_v2 = preprocess.preprocess_data(df, df_target)

    print("Creating Test Dataframe.")
    df_predict = preprocess.preprocess_predict_data(df_target, df_v1, df_v2)

    dataframe_list = [df_train, df_predict]
    dataframe_names = ["df_train", "df_predict"]

    print("Saving preprocessed data.")
    for data, name in zip(dataframe_list, dataframe_names):
        pickle.dump(data,
                    open(
                        config.processed_data_path + "/{}.pickle".format(
                            name),
                        "wb")
                    )


def deploy_model(df_train_path: str = config.DF_TRAIN_PATH):
    """
    Deploys the model.
    Parameters
    ----------
    df_train_path : Path for train data preprocessed.

    Returns
    -------

    """

    df = pickle.load(open(df_train_path, "rb"))

    # Model Stage
    print("Starting Model Stage")
    model = modelling.RegressorTrainer(
        df.drop(["V1", "V2"], axis=1),
        "prob_V1_V2",
        "Training Stage"
    )

    print("Setting up Pycaret Environment")
    model.start_session()

    print("Training The Model")
    model.train_model()

    print("Finalizing The Model")
    model.finalize_model()

    print("Generated Model Saved at: {}".format(config.models_path))
    model.save_model(config.models_path, "/ridge_model")


def predict_model(df_predict_path: str = config.DF_PREDICT_PATH, target: str = "prob_V1_V2", validation: bool = False):
    """
    Predicts data.
    Parameters
    ----------
    df_predict_path : Unseed preprocessed data.
    target : Target variable. For validation only.
    validation : Exports metrics if True. For validation only.

    Returns
    -------

    """

    predict = pickle.load(open(df_predict_path, "rb"))

    model = modelling.RegressorTrainer(
        predict.drop(["V1", "V2"], axis=1),
        "prob_V1_V2",
        "Prediction Stage"
    )
    model.load_model(config.models_path + "/ridge_model")

    if validation:
        prediction = model.predict_model(predict, target, export_metrics=True)
    else:
        prediction = model.predict_model(predict)

    pickle.dump(prediction, open(config.models_path + "/prediction.pickle", "wb"))
    print("Prediction Stage is Done.")


def run():
    """
    Run all model pipeline steps sequentially.
    :return:
    """
    features()
    deploy_model()
    predict_model()


def cli():
    """ Caller to transform module in a low-level CLI """
    return fire.Fire()


if __name__ == "__main__":
    cli()
