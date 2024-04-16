import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from boruta import BorutaPy

from typing import Union, List


class Feature_Importance:
    """
    A class for computing and visualizing feature importances.

    Args:
        model: The trained machine learning model.
        feature_names (List[str]): Names of the features (X).

    Attributes:
        model: The trained machine learning model.
        feature_names (List[str]): Names of the features .

    Methods:
        fi_impurity(): Computes feature importances using impurity-based methods.
        fi_impurity_visualize(std: np.array, forest_importances: pd.Series, **vis_params: dict): 
            Visualizes impurity-based feature importances.
        boruta_importance(X_train: Union[pd.DataFrame, np.array], y_train: Union[pd.DataFrame, np.array], **boruta_params: dict): 
            Computes feature importances using Boruta algorithm.
    """
    def __init__(self, model, feature_names: List[str]) -> None:
        """
        Initializes a Feature_Importance object.

        Args:
            model: The trained machine learning model.
            feature_names (List[str]): Names of the features.
        """
        self.model = model
        self.feature_names: List[str] = feature_names

    def fi_impurity(self) -> tuple:
        """
        Computes feature importances using impurity-based methods.

        Returns:
            tuple: A tuple containing standard deviations of feature importances and feature importances.
        """
        forest = self.model

        importances: np.array = forest.feature_importances_
        std: np.array = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

        forest_importances: pd.Series = pd.Series(importances, index=self.feature_names)
        return std, forest_importances
    
    def fi_impurity_visualize(self, std: np.array, forest_importances: pd.Series, **vis_params: dict) -> None:
        """
        Visualizes impurity-based feature importances.

        Args:
            std (np.array): Standard deviations of feature importances.
            forest_importances (pd.Series): Feature importances.
            **vis_params (dict): Additional parameters for visualization.

        Returns:
            None
        """
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax, **vis_params)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()

    def boruta_importance(self, X_train: Union[pd.DataFrame, np.array], y_train: Union[pd.DataFrame, np.array], **boruta_params: dict) -> BorutaPy:
        """
        Computes feature importances using Boruta algorithm.

        Args:
            X_train (Union[pd.DataFrame, np.array]): Training features.
            y_train (Union[pd.DataFrame, np.array]): Training target.
            **boruta_params (dict): Parameters for Boruta algorithm.

        Returns:
            BorutaPy: Boruta selector object.
        """
        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_

        forest = self.model

        boruta_selector: BorutaPy = BorutaPy(forest, **boruta_params)

        boruta_selector.fit(X_train, y_train)

        feature_ranks: list = list(zip(self.feature_names, 
                                boruta_selector.ranking_, 
                                boruta_selector.support_))
        
        for feat in feature_ranks:
            print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2])) 
        
        return boruta_selector