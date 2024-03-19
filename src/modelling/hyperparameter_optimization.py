from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
import hyperopt
from sklearn.metrics import root_mean_squared_error

import numpy as np
import pandas as pd

from typing import Union, Callable


class HyperparameterOpt:
    """
    Class for hyperparameter optimization using Hyperopt library.

    Args:
        model (class): Model class to be optimized.
        space (dict): Dictionary specifying the search space for hyperparameters.
        X_train (Union[pd.DataFrame, np.array]): Training features.
        y_train (Union[pd.DataFrame, np.array]): Training target variable.
        X_test (Union[pd.DataFrame, np.array]): Testing features.
        y_test (Union[pd.DataFrame, np.array]): Testing target variable.
        add_model_params (dict, optional): Additional model parameters. Default is None.

    Attributes:
        model (class): Model class to be optimized.
        space (dict): Dictionary specifying the search space for hyperparameters.
        best_params (dict): Best hyperparameters found during optimization.
        X_train (Union[pd.DataFrame, np.array]): Training features.
        y_train (Union[pd.DataFrame, np.array]): Training target variable.
        X_test (Union[pd.DataFrame, np.array]): Testing features.
        y_test (Union[pd.DataFrame, np.array]): Testing target variable.

    Methods:
        objective(params: dict) -> dict: Objective function to be optimized.
        hyperopt(algo: Callable=tpe.suggest, max_evals: int=100) -> dict: Perform hyperparameter optimization.

    """
    def __init__(self, model, space: dict, X_train: Union[pd.DataFrame, np.array], y_train: Union[pd.DataFrame, np.array], X_test: Union[pd.DataFrame, np.array], y_test: Union[pd.DataFrame, np.array], add_model_params: dict=None) -> None:
        """
        Initialize the HyperparameterOpt object.

        Args:
            model (class): Model class to be optimized.
            space (dict): Dictionary specifying the search space for hyperparameters.
            X_train (Union[pd.DataFrame, np.array]): Training features.
            y_train (Union[pd.DataFrame, np.array]): Training target variable.
            X_test (Union[pd.DataFrame, np.array]): Testing features.
            y_test (Union[pd.DataFrame, np.array]): Testing target variable.
            add_model_params (dict, optional): Additional model parameters. Default is None.

        Returns:
            None

        """
        self.model = model
        self.add_model_params: dict = add_model_params

        self.space: dict = space
        self.best_params: dict = None

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def objective(self, params: dict) -> dict:
        """
        Objective function to be optimized.

        Args:
            params (dict): Dictionary containing hyperparameters.

        Returns:
            dict: Dictionary containing the loss value and optimization status.

        """
        model = self.model(**params, **self.add_model_params)        
        model.fit(self.X_train, self.y_train)
        pred: np.array = model.predict(self.X_test)
        rmse: np.float64 = root_mean_squared_error(self.y_test, pred)
        return {'loss': rmse, 'status': STATUS_OK}
    
    def hyperopt(self, algo: Callable=tpe.suggest, max_evals: int=100) -> dict:
        """
        Perform hyperparameter optimization.

        Args:
            algo (Callable, optional): Optimization algorithm. Default is tpe.suggest.
            max_evals (int, optional): Maximum number of evaluations. Default is 100.

        Returns:
            dict: Best hyperparameters found during optimization.

        """
        trials: hyperopt.Trials = Trials()
        best: dict = fmin(fn=self.objective,
            space=self.space,
            algo=algo,
            max_evals=max_evals, 
            trials=trials
        )

        self.best_params: dict = space_eval(self.space, best)
        return self.best_params
