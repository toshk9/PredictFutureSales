import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from boruta import BorutaPy

class Feature_Importance:
    def __init__(self, model, X: pd.DataFrame, y) -> None:
        self.model = model
        self.X = X
        self.y = y
        
    def fi_impurity(self) -> tuple:
        feature_names = self.X.columns
        forest = self.model

        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

        forest_importances = pd.Series(importances, index=feature_names)
        return std, forest_importances
    
    def fi_impurity_visualize(self) -> None:
        std, forest_importances = self.fi_impurity()

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()

    def boruta_importance(self):
    
        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_

        forest = self.model

        boruta_selector = BorutaPy(forest, n_estimators='auto', verbose=2)


        boruta_selector.fit(self.X, self.y)

        feature_importance = boruta_selector.feature_importances_

        feature_names = np.array(self.X.columns)

        sorted_idx = np.argsort(feature_importance)[::-1]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance[sorted_idx], align='center')
        plt.xticks(range(len(feature_importance)), feature_names[sorted_idx], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.show()