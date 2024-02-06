import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest
import seaborn as sns
import numpy as np
import re
import os
import sqlite3
from typing import List, Tuple


class ETL:
    """
    ETL DOCSTRING
    """
    def __init__(self, data_path: str) -> None:
        """
        ETL INIT DOCSTRING
        """

        self.df: pd.DataFrame = pd.read_csv(data_path)
        self.dqc: DQC = DQC(self.df)
        
        print(f"{self.df.shape[0]} rows and {self.df.shape[1]} columns has been read from {os.path.basename(data_path)}")
    
    def get_data(self) -> pd.DataFrame:

        """
        Retrieves the transformed DataFrame.

        This method returns the DataFrame that has been transformed based on the selected options.

        :return: Transformed DataFrame.
        """
        return self.df

    def load_data_csv(self, file_name: str) -> None:

        """
        Saves the transformed DataFrame to a CSV file.

        The method stores the transformed DataFrame into a new CSV file at the specified location.

        :param file_name: The name of the CSV file to save.
        :return: None
        """

        self.df.to_csv('/data/processed/' + file_name + '.csv')
        print(f"File {file_name}.csv was successfully saved with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
    
    def load_data_sqlite(self, file_name: str) -> None:
        with sqlite3.connect(f"data/processed/{file_name}.db") as db: 
            self.df.to_sql(file_name, db, index=False, if_exists="replace")
        print(f"File {file_name}.db was successfully saved with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

    # DATA PROCESSING
    def outliers_processing(self, columns_to_process: List[str]) -> pd.DataFrame:
        outliers_check: dict = self.dqc.outliers_check(columns_to_process)["outliers_idxs"]
        for column in outliers_check.keys():
            self.df.loc[outliers_check[column], column] = self.df[column].mean()
            print(f"{len(outliers_check[column])} outliers in column {column} processed.")
        return self.df

    def na_processing(self) -> pd.DataFrame:
        na_values_idxs: pd.core.indexes.numeric.Int64Index = self.dqc.na_values_check()
        self.df.drop(index=na_values_idxs, inplace=True)
        print(f"{len(na_values_idxs)} rows with N/A values processed.")
        return self.df
    
    def inconst_dupl_processing(self, consistency_columns: List[str]) -> pd.DataFrame:
        inconsistent_duplicated_idxs: pd.core.indexes.numeric.Int64Index = self.dqc.consistency_uniqueness_check(consistency_columns)
        self.df.drop(index=inconsistent_duplicated_idxs, inplace=True)
        print(f"{len(inconsistent_duplicated_idxs)} inconsistent or duplicated rows processed.")
        return self.df

    def data_type_conversion(self) -> pd.DataFrame:
        # conversion numeric columns with not numeric data type 
        not_numeric_columns, not_date_columns, not_string_columns = self.dqc.types_check()

        for column_name in not_numeric_columns:
            try:
                self.df[column_name] = self.df[column_name].apply(np.int64)
            except Exception as e:
                raise e
        # conversion date columns with not date data type         
        for column_name in not_date_columns:
            try:
                self.df[column_name] = pd.to_datetime(self.df[column_name])
            except Exception as e:
                raise e
        # conversion string columns with not string data type         
        for column_name in not_string_columns:
            try:
                self.df[column_name] = self.df[column_name].apply(str)
            except Exception as e:
                raise e

        # converting columns of different data types to the columns of the same data type (data validity)
        for column in self.df.columns:
            if self.df.dtypes[column] == object: # we check columns with an object type, which could potentially be a column with data of different types
                
                expected_type = type(self.df[column].iloc[0]) # take the value type of the first row
                try:
                    self.df = self.df[column].apply(expected_type) # try to convert column samples with a wrong type into the right type (expected_type)
                except Exception as e:
                    raise e
        print("DataFrame column types processed.")
        return self.df

    def transform(self, outliers_columns: List[str], inconsistency_columns: List[str]) -> pd.DataFrame:
        transformations: list = [self.outliers_processing(outliers_columns), self.na_processing(), self.inconst_dupl_processing(inconsistency_columns), self.data_type_conversion()]
        for transformation in transformations:
            self.df = transformation

        return self.df

class DQC:
    """
    Data Quality Control (DQC) class for performing various data cleaning and quality assurance operations
    on a given DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:

        """
        Initializes the DQC class with the provided DataFrame.

        :param df: The DataFrame to perform quality control operations on.
        """

        self.df: pd.DataFrame = df

    # Basic functions
    def get_data(self) -> pd.DataFrame:

        """
        Retrieves the DataFrame.

        :return: The DataFrame.
        """

        return self.df

    def data_review(self):
        print("\nDataset Info:\n")
        self.df.info()
        print("\n", "-" * 50)

        print("\nDescriptive data statistics:\n")
        self.df.describe()
        print("\n", "-" * 50)

        print("\nPiece of the dataset:\n")
        self.df.head()
        print("\n", "-" * 50)

    def outliers_check(self, columns_to_check : List[str]) -> dict:
        columns_outliers: dict = {"outliers_idxs" : {}, "iqr_interval":{}} # dictionary for storing outliers indexes (for processing) 
                                                            # and iqr interval borders (for visualization)
        for column in columns_to_check:
            # iqr interval calculating 
            column_quartile1: float = self.df[column].quantile(.25)
            column_quartile3 = self.df[column].quantile(.75)
            column_iqr = column_quartile3 - column_quartile1

            column_interval_border1 = column_quartile1 - 1.5 * column_iqr
            column_interval_border2 = column_quartile3 + 1.5 * column_iqr
            columns_outliers["iqr_interval"][column] = (column_interval_border1, column_interval_border2)
            # outliers search
            outliers_idxs = self.df.loc[(self.df[column] < column_interval_border1) | (self.df[column] > column_interval_border2)].index
            columns_outliers["outliers_idxs"][column] = outliers_idxs

            print(f"{len(outliers_idxs)} outliers were found for the {column} column.")
        
        return columns_outliers

    def na_values_check(self):
        na_rows = self.df[self.df.isna().any(axis=1)] # rows with NA values 
        na_rows_idx = na_rows.index # rows indexes with NA values 
        
        print(f"Number of rows with missing values: {len(na_rows_idx)}")
        return na_rows_idx
    
    def consistency_uniqueness_check(self, consistency_columns: List[str]):
        for column in consistency_columns:
            if column not in self.df.columns:
                raise f"The specified column {column} does not exist in the Dataset"

        columns_to_consistency_check: List[str] = [column for column in self.df.columns if column not in consistency_columns]

        ds_without_the_feature = self.df.loc[:, columns_to_consistency_check]
        inconsistent_rows = ds_without_the_feature[ds_without_the_feature.duplicated()] # searching for identical rows, but with different values for the column item_cnt_day
        inconsistent_rows_idx = inconsistent_rows.index # indexes of identical rows

        print(f"Number of conflicting and duplicated rows: {len(inconsistent_rows_idx)}")
        return inconsistent_rows_idx

    def types_check(self) -> Tuple(List[str], List[str], List[str]):

        numeric_patterns = [r"(?:\w+|\b)id\b", r"(?:\w+|\b)price\b", r"(?:\w+|\b)num\b", r"(?:\w+|\b)cnt(?:\w+|\b)", r"(?:\w+|\b)amount(?:\w+|\b)"]
        date_patterns = [r"(?:\w+|\b)date\b"]
        string_patterns = [r"(?:\w+|\b)name\b"]

        not_numeric_columns: List[str] = []
        not_date_columns: List[str] = []
        not_string_columns: List[str] = []

        for column_name in self.df.columns:
            numeric_patterns_check = [bool(re.search(pattern, column_name, flags=re.IGNORECASE)) for pattern in numeric_patterns] 
            date_patterns_check = [bool(re.search(pattern, column_name, flags=re.IGNORECASE)) for pattern in date_patterns]
            string_patterns_check = [bool(re.search(pattern, column_name, flags=re.IGNORECASE)) for pattern in string_patterns]

            if any(numeric_patterns_check):
                numeric_types = [int, float, np.int64, np.float64]

                numeric_check = [self.df.dtypes[column_name] == ntype for ntype in numeric_types]
                if not any(numeric_check):
                    not_numeric_columns.append(column_name)

            elif any(date_patterns_check):
                if not pd.api.types.is_datetime64_any_dtype(self.df[column_name]):
                    not_date_columns.append(column_name)

            elif any(string_patterns_check):
                if not self.df.dtypes[column_name] == object:
                    not_string_columns.append(column_name)
                
        print(f"Сolumns with presumably the wrong data type: \nnot numeric type columns: {not_numeric_columns},\nnot date type columns: {not_date_columns},\nnot string type columns: {not_string_columns}")
        return not_numeric_columns, not_date_columns, not_string_columns

class Visualization