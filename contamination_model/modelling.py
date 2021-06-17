import json
import pandas as pd
import pycaret.regression as pcr
from contamination_model import config
from pycaret.utils import check_metric


def evaluation_metrics(df: pd.DataFrame, target: str, export_metrics: bool = False) -> None:
    """

    Parameters
    ----------
    df :  Validation dataframe.
    target : The dataframe's target variable.
    export_metrics : If true, generates and export performance metrics.

    Returns
    -------

    """

    metric_values = []
    print('Model Evaluation Performance:')

    for metric in config.metric_list:
        value = round(check_metric(df[target], df['Label'], metric=metric), 2)
        metric_values.append(value)

    evaluation = dict(zip(config.metric_list, metric_values))
    print(evaluation)

    if export_metrics:
        with open(config.models_path + "/evaluation.json", "w") as file:
            json.dump(evaluation, file)


class RegressorTrainer:
    def __init__(self, df: pd.DataFrame, target: str, exp_name: str, session_id: int = 16):
        """
        Initialize classe objects.
        Parameters
        ----------
        df : Cleaned dataframe for model ingestion.
        target : Target variable.
        exp_name : Model experiment name, for mlflow tracking purposes.
        session_id : experiment's random state.
        """
        self.df = df
        self.target = target
        self.exp_name = exp_name
        self.session_id = session_id
        self.categorical_features = self.df.select_dtypes(exclude=["int64", "float64"]).columns.to_list()

    def start_session(self):
        """
        Starts pycaret environment.
        Returns
        -------
        None

        """
        print("Setting Up PyCaret Session for Machine Learning Experiments")
        pcr.setup(
            data=self.df,
            target=self.target,
            experiment_name=self.exp_name,
            categorical_features=self.categorical_features,
            session_id=self.session_id,
            normalize=True,
            silent=True,
            verbose=False
        )

    def train_model(self, model: str = "ridge"):
        """
        Train and optimize model.
        Returns
        -------
        Final model.
        """

        print("Training Ridge Model")
        self.model = pcr.create_model(model,
                                      verbose=False)
        print("Training Finished")

        self.metrics = pcr.pull()
        cross_valid_train_metrics = self.metrics.loc["Mean"]

        print("Model's Metrics:")
        print(cross_valid_train_metrics)

    def finalize_model(self):
        """
        This function fits the estimator onto the complete
        dataset passed during the setup() stage.
        The purpose of this function is to
        prepare for final model deployment after experimentation.
        Description taken from pycaret/classification.

        Returns
        -------
        Trained model object fitted on complete dataset.
        """
        pcr.finalize_model(self.model)

    def save_model(self, path: str, model_name: str):
        """
        Saves model to path.
        Parameters
        ----------
        path : Path to save model.
        model_name: Model Name.
        Returns
        -------
        None
        """

        pcr.save_model(self.model, path + model_name)

    def load_model(self, path: str):
        """
        Loads the model.
        Parameters
        ----------
        path : Model's path.

        Returns
        -------
        Model
        """

        self.model = pcr.load_model(path)

    def predict_model(self, data: pd.DataFrame, target: str = None, export_metrics: bool = False):
        """
        Make predictions with unseen data.
        Parameters
        ----------
        data : The new and unseen data for predictions.
        target: Target variable for validation evaluation.
        export_metrics : Checks whether model performance metrics are exported. Validation datasets Only.

        Returns
        -------
        pd.DataFrame

        """

        predict = pcr.predict_model(estimator=self.model, data=data)
        if export_metrics:
            evaluation_metrics(predict, target, export_metrics)

        return predict

    @property
    def get_model(self):
        return self.model
