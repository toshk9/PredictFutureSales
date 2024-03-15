from tqdm.notebook import tqdm as notebook_tqdm

import shap
import numpy as np
import matplotlib.pyplot as plt


class Explainability_layer:
    def __init__(self, model, X_train, X_test) -> None:
        self.model = model

        self.X_train = X_train
        self.X_test = X_test

    def get_shap_values(self):

        explainer = shap.Explainer(self.model, self.X_train)

        shap_values = explainer.shap_values(self.X_test)
        return shap_values
    
    def fi_shap_visualization(self):
        shap_values = get_shap_values(X_train, X_test)

        shap.summary_plot(shap_values, X_test)

    # Отображение SHAP значений для конкретного предсказания (например, для первого элемента в тестовом наборе)
        shap.initjs()
        shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[0,:])

        # Визуализация важности признаков
        shap.summary_plot(shap_values, X_test)

    # Визуализация зависимости SHAP значений от значений признаков
    shap.dependence_plot("item_id", shap_values, X_test)