from tqdm.notebook import tqdm as notebook_tqdm

import shap

import numpy as np
import pandas as pd

from typing import List, Union


class Explainability_layer:
    """
    A class for explaining model predictions using SHAP (SHapley Additive exPlanations).

    Args:
        model: The trained machine learning model.
        X_train (Union[np.array, pd.DataFrame]): Training features.
        X_test (Union[np.array, pd.DataFrame]): Testing features.
        feature_names (List[str], optional): Names of the features.

    Attributes:
        model: The trained machine learning model.
        X_train (np.array): Training features.
        X_test (np.array): Testing features.
        feature_names (Union[pd.Index, List[str]]): Names of the features.
        explainer (shap.Explainer): SHAP explainer object.
        shap_values (np.array): SHAP values.

    Methods:
        calc_shap_values(): Calculates SHAP values for the testing features.
        shap_visualization(dep_feature_name: str, dep_interaction_feature_name: str, force_sample_idx: int=0): 
            Visualizes SHAP values.
    """
    def __init__(self, model, X_train: Union[np.array, pd.DataFrame], X_test: Union[np.array, pd.DataFrame], feature_names: List[str]=None) -> None:
        """
        Initializes an Explainability_layer object.

        Args:
            model: The trained machine learning model.
            X_train (Union[np.array, pd.DataFrame]): Training features.
            X_test (Union[np.array, pd.DataFrame]): Testing features.
            feature_names (List[str], optional): Names of the features.
        """
        
        self.model = model

        self.X_train: np.array = X_train
        self.X_test: np.array = X_test

        if type(X_train) == pd.DataFrame:
            try:
                self.feature_names: pd.Index = X_train.columns
            except:
                self.feature_names: List[str] = feature_names
        else:
            self.feature_names: List[str] = feature_names

        self.explainer: shap.Explainer = shap.Explainer(self.model, self.X_train)
        self.shap_values: np.array = None

        shap.initjs()

    def calc_shap_values(self) -> np.array:
        """
        Calculates SHAP values for the testing features.

        Returns:
            np.array: SHAP values.
        """
    
        self.shap_values: np.array = self.explainer.shap_values(self.X_test)
        return self.shap_values
    
    def shap_visualization(self, dep_feature_name: str, dep_interaction_feature_name: str, force_sample_idx: int=0) -> None:
        """
        Visualizes SHAP values.

        Args:
            dep_feature_name (str): Name of the dependent feature.
            dep_interaction_feature_name (str): Name of the interaction feature.
            force_sample_idx (int, optional): Index of the sample for force plot. Defaults to 0.

        Returns:
            None
        """
        if self.shap_values is None:
            self.shap_values: np.array = self.calc_shap_values()

        shap.summary_plot(self.shap_values, self.X_test, feature_names=self.feature_names)
        print("\nForce plot:\n")
        shap.force_plot(self.explainer.expected_value, self.shap_values[force_sample_idx,:], self.X_test[force_sample_idx,:], feature_names=self.feature_names)
        print("\Dependence plot:\n")
        shap.dependence_plot(ind=dep_feature_name, shap_values=self.shap_values, features=self.X_test, interaction_index=dep_interaction_feature_name, feature_names=self.feature_names)