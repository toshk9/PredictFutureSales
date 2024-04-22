from typing import Union, List
from tensorflow_metadata.proto.v0 import statistics_pb2
import argparse

import pandas as pd
import numpy as np

from data.etl_layer import ETL

from modelling.feature_extraction import Feature_Extraction as FE
from modelling.data_validation import Data_Validation as DV

from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor


class Model:
    """
    A class representing a machine learning model.

    Attributes:
        model: The CatBoostRegressor model with predefined categorical features.
        model_predictions: Predictions made by the model.

    Methods:
        get_train_test_data(raw_data_path, test_size): Loads and preprocesses training data, and splits it into train and test sets.
        model_inference(inference_data): Performs inference using the model on new data.
        save_model_predictions(file_title): Saves model predictions to a CSV file.
    """
    def __init__(self) -> None:
        """
        Initializes a Model object with a pre-trained CatBoostRegressor model.
        """
        self.model = CatBoostRegressor(cat_features=["shop_id", "item_id", "month_num"])
        self.model.load_model("models/CatBoost")
        self.model_predictions: np.array = None

    def get_train_test_data(self, raw_data_path:str = "data/raw/sales_train.csv", test_size: np.float64=0.3) -> tuple:
        """
        Loads and preprocesses training data, and splits it into train and test sets.

        Args:
            raw_data_path (str): Path to the raw sales data CSV file.
            test_size (np.float64): Proportion of the dataset to include in the test split.

        Returns:
            tuple: Tuple containing X_train, X_test, y_train, y_test, and feature_names.
        """
        sales_train_etl: ETL = ETL(raw_data_path)
        sales_train_etl.transform(["item_cnt_day"], ["item_cnt_day"])
        sales_train_df: pd.DataFrame = sales_train_etl.get_data()

        full_processed_monthly_df: pd.DataFrame = pd.read_csv("data/processed/monthly_sales_full_processed.csv", parse_dates=["month"])

        train_df: pd.DataFrame = FE(sales_train_df, full_processed_monthly_df).get_fe_df()

        train_etl: ETL = ETL(df=train_df)
        train_etl.transform(["item_cnt_month"], ["item_cnt_month"])
        train_df: pd.DataFrame = train_etl.get_data()
        
        X: np.array = train_df.drop(["item_cnt_month"], axis=1).values
        y: np.array = train_df[["item_cnt_month"]].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        feature_names: List[str] = train_df.columns
        return X_train, X_test, y_train, y_test, feature_names
    
    def model_inference(self, inference_data: pd.DataFrame) -> np.array:
        """
        Performs inference using the model on new data.

        Args:
            inference_data (pd.DataFrame): New raw data for inference.

        Returns:
            np.array: Predictions made by the model.
        """
        inference_data.reset_index(drop=True, inplace=True)

        i_data_dv: DV = DV()
        i_data_stats: statistics_pb2.DatasetFeatureStatisticsList = i_data_dv.create_stats(inference_data)

        raw_data_dv: DV = DV()
        raw_data_dv.load_schema("data/data_validation/raw_data_schema.pbtxt")
        raw_data_dv.load_stats("data/data_validation/raw_data_stats.txt")

        raw_validate_stats = raw_data_dv.validate_data(i_data_stats)

        if raw_validate_stats.anomaly_info:
            raise ValueError("Anomalies in statistics have been found in the new raw data.")
    
        fe_i_data = FE(inference_data).get_fe_df()

        fe_i_data_dv: DV = DV()
        fe_i_data_stats: statistics_pb2.DatasetFeatureStatisticsList = fe_i_data_dv.create_stats(fe_i_data)

        train_data_dv: DV = DV()
        train_data_dv.load_schema("data/data_validation/train_data_schema.pbtxt")
        train_data_dv.load_stats("data/data_validation/train_data_stats.txt")

        train_validate_stats = train_data_dv.validate_data(fe_i_data_stats)

        if train_validate_stats.anomaly_info:
            raise ValueError("Anomalies in statistics have been found in the new processed data.")

        model_predictions: np.array = self.model.predict(fe_i_data)
        self.model_predictions: np.array = model_predictions
        return model_predictions

    def save_model_predictions(self, file_title:str)-> None:
        """
        Saves model predictions to a CSV file.

        Args:
            file_title (str): Title for the CSV file.

        Returns:
            None
        """
        model_predictions: pd.DataFrame = pd.DataFrame(self.model_predictions, columns=["item_cnt_month"])
        model_predictions.reset_index(inplace=True)
        model_predictions.rename(columns={'index': 'ID'}, inplace=True)
        model_predictions.to_csv(f"data/model_predictions/{file_title}.csv", index=False)
        print("\nThe results of the model prediction have been successfully saved.")


if __name__ == "__main__":
    m = Model()
    parser = argparse.ArgumentParser(description='Run model inference from the command line.')
    parser.add_argument('--inference', type=str, help='Please specify the directory of the raw data file for the inference (the data must match a predefined schema)', required=True)
    args = parser.parse_args()

    inference_data = ETL(args.inference).get_data()
    m.model_inference(inference_data)
    print("The results of the model have been successfully obtained!\nSave?")
    save_choice = input("[Y/N] : ")
    if save_choice == "Y":
        save_file_title = input("Please specify the file title to save model prediction results: ")
        m.save_model_predictions(save_file_title)



