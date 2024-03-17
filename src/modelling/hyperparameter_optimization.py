from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
import hyperopt
from sklearn.metrics import root_mean_squared_error

import numpy as np
import pandas as pd

from typing import Union


class HyperparameterOpt:
    """
    A class for hyperparameter optimization using Hyperopt library.

    Args:
        model: The machine learning model class.
        space (dict): Hyperparameter search space.

    Attributes:
        model: The machine learning model class.
        space (dict): Hyperparameter search space.
        best_params (dict): Best hyperparameters found during optimization.

    Methods:
        objective(params: dict, add_model_params: dict, X_train: Union[pd.DataFrame, np.array], y_train: Union[pd.DataFrame, np.array], X_test: Union[pd.DataFrame, np.array], y_test: Union[pd.DataFrame, np.array]) -> dict: 
            Objective function for hyperparameter optimization.
        hyperopt(algo: function=tpe.suggest, max_evals: int=100): 
            Performs hyperparameter optimization using Hyperopt.
    """
    def __init__(self, model, space: dict):
        """
        Initializes a HyperparameterOpt object.

        Args:
            model: The machine learning model class.
            space (dict): Hyperparameter search space.
        """
        self.model = model
        self.space: dict = space
        self.best_params: dict = None

    def objective(self, params: dict, add_model_params: dict, X_train: Union[pd.DataFrame, np.array], y_train: Union[pd.DataFrame, np.array], X_test: Union[pd.DataFrame, np.array], y_test: Union[pd.DataFrame, np.array]) -> dict:
        """
        Objective function for hyperparameter optimization.

        Args:
            params (dict): Hyperparameters to evaluate.
            add_model_params (dict): Additional model parameters.
            X_train (Union[pd.DataFrame, np.array]): Training features.
            y_train (Union[pd.DataFrame, np.array]): Training target.
            X_test (Union[pd.DataFrame, np.array]): Testing features.
            y_test (Union[pd.DataFrame, np.array]): Testing target.

        Returns:
            dict: Dictionary containing loss and status.
        """
        model = self.model(**params, **add_model_params)        
        model.fit(X_train, y_train)
        pred: np.array = model.predict(X_test)
        rmse: np.float64 = root_mean_squared_error(y_test, pred)
        return {'loss': rmse, 'status': STATUS_OK}
    
    def hyperopt(self, algo: function=tpe.suggest, max_evals: int=100):
        """
        Performs hyperparameter optimization using Hyperopt.

        Args:
            algo (function, optional): Hyperopt algorithm. Defaults to tpe.suggest.
            max_evals (int, optional): Maximum number of evaluations. Defaults to 100.

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
