import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

class ModelErrorAnalysis:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.predictions = self.model.predict(self.X)
        self.errors = self.predictions - self.y

    def calculate_metrics(self):
        mae = np.mean(np.abs(self.errors))
        mse = mean_squared_error(self.y, self.predictions)
        rmse = np.sqrt(mse)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

    def plot_residuals(self):
        plt.figure(figsize=(10, 6))
        sns.residplot(self.predictions, self.errors, lowess=True, line_kws={'color': 'red', 'lw': 1})
        plt.title('Residuals Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()

    def analyze_big_target(self, target_threshold):
        big_target_indices = self.y >= target_threshold
        big_target_errors = self.errors[big_target_indices]
        big_target_mae = np.mean(np.abs(big_target_errors))
        return big_target_mae

    def analyze_small_dynamic(self, dynamic_threshold):
        dynamic_indices = np.abs(self.y) <= dynamic_threshold
        dynamic_errors = self.errors[dynamic_indices]
        dynamic_mae = np.mean(np.abs(dynamic_errors))
        return dynamic_mae

    def find_influential_samples(self, threshold):
        influential_samples = np.abs(self.errors) > threshold
        return self.X[influential_samples], self.y[influential_samples], self.errors[influential_samples]

# Example usage:
# model_error_analysis = ModelErrorAnalysis(model, X_test, y_test)
# metrics = model_error_analysis.calculate_metrics()
# print("Model Metrics:", metrics)
# model_error_analysis.plot_residuals()
# big_target_mae = model_error_analysis.analyze_big_target(100)
# print("MAE for big targets:", big_target_mae)
# dynamic_mae = model_error_analysis.analyze_small_dynamic(10)
# print("MAE for small dynamic:", dynamic_mae)
# influential_X, influential_y, influential_errors = model_error_analysis.find_influential_samples(20)
# print("Influential Samples:", influential_X, influential_y, influential_errors)
