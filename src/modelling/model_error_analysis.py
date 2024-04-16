import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

from typing import Union


class ModelErrorAnalysis:
    """
    A class for analyzing errors of a machine learning model.

    Args:
        model: The trained machine learning model.
        X_test (Union[pd.DataFrame, np.array]): Testing features.
        y_test (Union[pd.DataFrame, np.array]): Testing target.

    Attributes:
        model: The trained machine learning model.
        X_test (Union[pd.DataFrame, np.array]): Testing features.
        y_test (Union[pd.DataFrame, np.array]): Testing target.
        predictions (np.array): Predictions made by the model.
        errors (np.array): Errors between predictions and actual values.

    Methods:
        calculate_metrics(): Calculates error metrics.
        plot_residuals(): Plots residuals.
        analyze_big_target(target_threshold): Analyzes errors for large target values.
        analyze_small_target(target_threshold): Analyzes errors for small target values.
        find_influential_samples(threshold): Finds influential samples based on error threshold.
    """
    def __init__(self, model, X_test: Union[pd.DataFrame, np.array], y_test: Union[pd.DataFrame, np.array]) -> None:
        """
        Initializes a ModelErrorAnalysis object.

        Args:
            model: The trained machine learning model.
            X_test (Union[pd.DataFrame, np.array]): Testing features.
            y_test (Union[pd.DataFrame, np.array]): Testing target.
        """
        self.model = model
        self.X_test: Union[pd.DataFrame, np.array] = X_test
        self.y_test: Union[pd.DataFrame, np.array] = y_test

        self.predictions: np.array = self.model.predict(self.X_test)

        self.errors: np.array = self.predictions - self.y_test

    def calculate_metrics(self) -> dict:
        """
        Calculates error metrics.

        Returns:
            dict: Dictionary containing error metrics.
        """
        mae: np.float64 = np.mean(np.abs(self.errors))
        mse: np.float64 = mean_squared_error(self.y_test, self.predictions)
        rmse: np.float64 = np.sqrt(mse)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

    def plot_residuals(self) -> None:
        """
        Plots residuals.
        
        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        sns.residplot(x=self.predictions, y=self.errors, lowess=True, line_kws={'color': 'red', 'lw': 1})
        plt.title('Residuals Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()

    def analyze_big_target(self, target_threshold) -> np.float64:
        """
        Analyzes errors for large target values.

        Args:
            target_threshold: Threshold for defining large target values.

        Returns:
            np.float64: Mean absolute error for large target values.
        """
        big_target_indices: np.array = self.y_test >= target_threshold
        big_target_errors: np.array  = self.errors[big_target_indices]
        big_target_mae: np.float64 = np.mean(np.abs(big_target_errors))
        return big_target_mae

    def analyze_small_target(self, target_threshold) -> np.float64:
        """
        Analyzes errors for small target values.

        Args:
            target_threshold: Threshold for defining small target values.

        Returns:
            np.float64: Mean absolute error for small target values.
        """
        small_target_indices: np.array  = np.abs(self.y_test) <= target_threshold
        small_target_errors: np.array  = self.errors[small_target_indices]
        small_target_mae: np.float64 = np.mean(np.abs(small_target_errors))
        return small_target_mae

    def find_influential_samples(self, threshold) -> tuple:
        """
        Finds influential samples based on error threshold.

        Args:
            threshold: Error threshold for defining influential samples.

        Returns:
            tuple: Tuple containing influential samples' features, target values, and errors.
        """
        influential_samples: np.array = np.abs(self.errors) > threshold
        return self.X_test[influential_samples], self.y_test[influential_samples], self.errors[influential_samples]