import pandas as pd
import numpy as np

import re

from typing import List


class DQC:
    """
    Data Quality Control (DQC) class for performing various data cleaning and quality assurance operations
    on a given DataFrame.

    :param df: The DataFrame to perform quality control operations on.
    :param df_title: Optional title for the DataFrame.

    The DQC class allows users to conduct quality control checks on a DataFrame, 
    such as identifying missing values, detecting outliers, assessing data consistency, 
    and checking data types. Once initialized, users can utilize various methods provided 
    by the class to analyze and improve the quality of the data.
    """
    def __init__(self, df: pd.DataFrame, df_title: str=None) -> None:
        """
        Initializes the DQC class with the provided DataFrame.

        :param df: The DataFrame to perform quality control operations on.
        :param df_title: Optional title for the DataFrame.

        This method initializes an instance of the DQC class with the provided DataFrame. 
        It assigns the DataFrame to the instance variable 'df' and optionally assigns a 
        title to 'df_title'. If no title is provided, it defaults to None.
        """

        self.df: pd.DataFrame = df
        self.df_title: str = df_title

    def get_data(self) -> pd.DataFrame:
        """
        Retrieves the DataFrame.

        :return: The DataFrame.

        This method returns the DataFrame stored in the DQC instance. 
        It allows users to access the DataFrame for further analysis or manipulation.
        """
        return self.df

    def data_review(self) -> None:
        """
        Prints a review of the dataset.

        This method prints various information about the dataset, including:
        - Dataset title (if provided)
        - DataFrame info, which includes the data types, non-null counts, and memory usage
        - Descriptive statistics such as count, mean, standard deviation, min, and max values
        - A snippet of the dataset showing the first 5 rows

        It serves as a quick overview of the dataset's structure and contents.
        """
        print(f"\nDataset {self.df_title} info:\n")
        print(self.df.info())
        print("\n", "-" * 50)

        print("\nDescriptive data statistics:\n")
        print(self.df.describe())
        print("\n", "-" * 50)

        print("\nPiece of the dataset:\n")
        print(self.df.head(5))
        print("\n", "-" * 50)

    def outliers_check(self, columns_to_check : List[str]) -> dict:
        """
        Checks for outliers in specified columns using the Interquartile Range (IQR) method.

        :param columns_to_check: A list of column names in the DataFrame to check for outliers.
        :return: A dictionary containing information about outliers and their indexes.

        This method calculates the Interquartile Range (IQR) for each specified column 
        and identifies outliers based on the IQR boundaries. It returns a dictionary 
        containing two keys: 
        - "outliers_idxs": A dictionary mapping column names to lists of indexes 
        where outliers were found.
        - "iqr_interval": A dictionary mapping column names to tuples representing 
        the lower and upper bounds of the IQR interval for each column.
        """
        for column in columns_to_check:
            if column not in self.df.columns:
                raise f"The specified column {column} does not exist in the Dataset"
        
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

            print(f"\n{len(outliers_idxs)} outliers were found for the {column} column.")
        
        return columns_outliers

    def na_values_check(self) -> pd.Index:
        """
        Checks for rows with missing values in the DataFrame.

        :return: A pandas Int64Index containing the indexes of rows with missing values.

        This method identifies rows in the DataFrame that contain one or more missing values (NaNs). 
        It returns the indexes of these rows for further investigation or handling.
        """
        na_rows: pd.core.frame.DataFrame = self.df[self.df.isna().any(axis=1)] 
        na_rows_idx: pd.Index = na_rows.index

        print(f"\nNumber of rows with missing values: {len(na_rows_idx)}")
        return na_rows_idx
    
    def consistency_uniqueness_check(self, consistency_columns: List[str]) -> pd.Index:
        """
        Checks for consistency and uniqueness of specified columns in the DataFrame.

        :param consistency_columns: A list of column names to check for consistency and uniqueness.
        :return: A pandas Int64Index containing the indexes of rows with conflicting or duplicated data.

        This method checks for conflicting or duplicated rows in the DataFrame based on specified columns. 
        It returns the indexes of rows where inconsistencies or duplications are found.
        """
        
        for column in consistency_columns:
            if column not in self.df.columns:
                raise f"The specified column {column} does not exist in the Dataset"

        columns_to_consistency_check: List[str] = [column for column in self.df.columns if column not in consistency_columns]

        ds_without_the_feature = self.df.loc[:, columns_to_consistency_check]
        inconsistent_rows = ds_without_the_feature[ds_without_the_feature.duplicated()] 
        inconsistent_rows_idx = inconsistent_rows.index

        print(f"\nNumber of conflicting or duplicated rows: {len(inconsistent_rows_idx)}")
        return inconsistent_rows_idx

    def types_check(self) -> tuple:
        """
        Checks the data types of columns in the DataFrame.

        :return: A tuple containing lists of columns with potentially incorrect data types.

        This method examines the data types of columns in the DataFrame and identifies potential inconsistencies. 
        It returns three lists:
        - Columns that should be numeric but are not.
        - Columns that should be dates but are not recognized as such.
        - Columns that should be strings but are not recognized as such.
        """

        numeric_patterns = [r"(?:\w+|\b)id\b", r"(?:\w+|\b)price\b", r"(?:\w+|\b)num\b", r"(?:\w+|\b)cnt(?:\w+|\b)", r"(?:\w+|\b)amount(?:\w+|\b)"]
        date_patterns = [r"(?:\w+|\b)date\b", r"(?:\w+|\b)month\b"]
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
                
        print(f"\nСolumns with presumably the wrong data type: \nnot numeric type columns: {not_numeric_columns},\nnot date type columns: {not_date_columns},\nnot string type columns: {not_string_columns}")
        return not_numeric_columns, not_date_columns, not_string_columns