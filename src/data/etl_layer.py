import pandas as pd
import numpy as np

import sqlite3
import os

from typing import List

from .eda_layer import EDA
from .dqc_layer import DQC


class ETL:
    """
    A class for performing Extract, Transform, Load (ETL) operations on datasets.

    This class is designed to streamline the process of data extraction, transformation, and loading.
    It provides functionalities to read data from various sources, apply data cleaning and preprocessing techniques,
    and save the processed data to different formats.

    Attributes:
    -----------
    df : pd.DataFrame
        The pandas DataFrame holding the data.
    df_title : str
        The title or name of the dataset, used for identification purposes.
    dqc : DQC
        An instance of the Data Quality Check (DQC) class, used for performing data quality checks and preprocessing.

    eda : EDA
        An instance of the Exploratory Data Analysis (EDA) class, used for performing exploratory data analysis and preprocessing.    

    Methods:
    --------
    __init__(self, data_path: str=None, df: pd.DataFrame=None):
        Initializes the ETL object, loads data from a CSV file or uses an existing DataFrame.

    get_data(self) -> pd.DataFrame:
        Retrieves the currently processed DataFrame.

    load_data_csv(self, file_name: str) -> None:
        Saves the DataFrame to a CSV file at a specified location.

    load_data_sqlite(self, file_name: str) -> None:
        Saves the DataFrame to an SQLite database file.

    outliers_processing(self, columns_to_process: List[str]) -> pd.DataFrame:
        Processes outliers in specified columns by replacing them with the column mean.

    na_processing(self) -> pd.DataFrame:
        Drops rows with NaN values.

    inconst_dupl_processing(self, consistency_columns: List[str]) -> pd.DataFrame:
        Drops rows that are inconsistent or duplicated based on specified columns.

    data_type_conversion(self, dayfirst: bool=True) -> pd.DataFrame:
        Converts column data types to appropriate formats (e.g., strings to dates, objects to numerics).
    
    def ts_nonstatinarity_processing(self, column: str) -> pd.DataFrame:
        Processes a specified column in the DataFrame to achieve stationarity of time series.
        
    transform(self, outliers_columns: List[str], inconsistency_columns: List[str], dayfirst: bool=True) -> pd.DataFrame:
        Applies a series of transformations to the DataFrame, including outlier processing, NA processing, inconsistency and duplication handling, and data type conversion.
    """
    def __init__(self, data_path: str=None, df: pd.DataFrame=None) -> None:
        """
        Initializes the ETL object with either a path to a CSV file or an existing DataFrame.

        Parameters:
        -----------
        data_path : str, optional
            The path to the CSV file containing the data. If not specified, `df` must be provided.
        df : pd.DataFrame, optional
            An existing DataFrame to use. If not specified, data will be loaded from `data_path`.

        Raises:
        -------
        ValueError:
            If neither `data_path` nor `df` is provided.
        """

        if df is None:
            self.df: pd.DataFrame = pd.read_csv(data_path) 
            self.df_title: str = os.path.basename(data_path)
        else: 
            self.df: pd.DataFrame = df
            self.df_title: str = "Dataset" 

        self.dqc: DQC = DQC(self.df, self.df_title)
        self.eda: EDA = EDA(self.df)
        
        print(f"\n{self.df.shape[0]} rows and {self.df.shape[1]} columns has been read from {self.df_title}")
    
    def get_data(self) -> pd.DataFrame:
        """
        Retrieves the currently processed DataFrame.

        This method returns the DataFrame that has been subjected to various transformations and cleaning processes.
        It allows for the extraction of the dataset after all specified ETL operations have been performed,
        making it ready for analysis, visualization, or further processing.

        Returns:
        -------- 
        pd.DataFrame
            The transformed and cleaned pandas DataFrame currently held by the ETL instance.
        """
        return self.df

    def load_data_csv(self, file_name: str) -> None:
        """
        Saves the transformed DataFrame to a CSV file.

        This method stores the DataFrame into a new CSV file at the specified location.
        The file is saved with the specified `file_name` in the 'data/processed' directory.

        Parameters:
        -----------
        file_name : str
            The name of the CSV file to save. The file extension ".csv" will be automatically appended.

        Raises:
        -------
        FileNotFoundError:
            If the 'data/processed' directory does not exist.

        Notes:
        ------
        The CSV file will be saved in the 'data/processed' directory relative to the current working directory.
        """

        self.df.to_csv(f"../data/processed/{file_name}.csv")
        print(f"\nFile {file_name}.csv was successfully saved with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
    
    def load_data_sqlite(self, file_name: str) -> None:
        """
        Saves the transformed DataFrame to an SQLite database file.

        This method stores the DataFrame into a new SQLite database file at the specified location.
        The DataFrame is saved as a new table within the database with the specified `file_name`.
        If a table with the same name already exists, it will be replaced.

        Parameters:
        -----------
        file_name : str
            The name of the SQLite database file to save. The file extension ".db" will be automatically appended.

        Raises:
        -------
        sqlite3.Error:
            If an error occurs while connecting to or writing to the SQLite database.

        Notes:
        ------
        The SQLite database file will be saved in the 'data/processed' directory relative to the current working directory.
        """
        with sqlite3.connect(f"../data/processed/{file_name}.db") as db: 
            self.df.to_sql(file_name, db, index=False, if_exists="replace")
        print(f"\nFile {file_name}.db was successfully saved with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

    # DATA PROCESSING
    def outliers_processing(self, columns_to_process: List[str]) -> pd.DataFrame:
        """
        Process outliers in specified columns by replacing them with the column mean.

        This method identifies outliers in the specified columns using the Data Quality Check (DQC) instance.
        Outliers are replaced with the mean value of the corresponding column.

        Parameters:
        -----------
        columns_to_process : List[str]
            A list of column names in which to process outliers.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with outliers processed.

        Notes:
        ------
        Outliers are identified and replaced individually for each specified column.
        The mean value of each column is used as the replacement for outliers.
        """
        outliers_check: dict = self.dqc.outliers_check(columns_to_process)["outliers_idxs"]
        for column in outliers_check.keys():
            self.df.loc[outliers_check[column], column] = self.df[column].mean()
            print(f"\n{len(outliers_check[column])} outliers in column {column} processed.")
        return self.df

    def na_processing(self) -> pd.DataFrame:
        """
        Process rows with missing values (NaN) by dropping them from the DataFrame.

        This method identifies rows with missing values using the Data Quality Check (DQC) instance
        and removes them from the DataFrame.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with rows containing missing values removed.

        Notes:
        ------
        Rows containing missing values are dropped entirely from the DataFrame.
        The number of rows dropped is printed for reference.
        """
        na_values_idxs: pd.Index = self.dqc.na_values_check()
        self.df.drop(index=na_values_idxs, inplace=True)
        print(f"\n{len(na_values_idxs)} rows with N/A values processed.")
        return self.df
    
    def inconst_dupl_processing(self, consistency_columns: List[str]) -> pd.DataFrame:
        """
        Process inconsistent or duplicated rows based on specified columns.

        This method identifies inconsistent or duplicated rows based on the specified columns using the
        Data Quality Check (DQC) instance and removes them from the DataFrame.

        Parameters:
        -----------
        consistency_columns : List[str]
            A list of column names used to check for inconsistency or duplication.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with inconsistent or duplicated rows removed.

        Notes:
        ------
        Rows that are identified as inconsistent or duplicated based on the specified columns are dropped entirely
        from the DataFrame. The number of rows processed and removed is printed for reference.
        """
        inconsistent_duplicated_idxs: pd.Index = self.dqc.consistency_uniqueness_check(consistency_columns)
        self.df.drop(index=inconsistent_duplicated_idxs, inplace=True)
        print(f"\n{len(inconsistent_duplicated_idxs)} inconsistent or duplicated rows processed.")
        return self.df

    def data_type_conversion(self, dayfirst: bool=True) -> pd.DataFrame:
        """
        Convert DataFrame columns to appropriate data types.

        This method converts DataFrame columns to appropriate data types based on their content.
        Numeric columns with non-numeric data types are converted to integer type (int64).
        Date columns with non-date data types are converted to datetime objects.
        String columns with non-string data types are converted to string objects.

        Parameters:
        -----------
        dayfirst : bool, optional (default=True)
            Whether to interpret dates with the day first (e.g., '10/01/2022' as January 10, 2022).

        Returns:
        --------
        pd.DataFrame
            The DataFrame with columns converted to appropriate data types.

        Notes:
        ------
        Columns with different data types are converted to the same data type for data validity.
        The resulting DataFrame reflects the data type conversions performed.
        """
        # conversion numeric columns with not numeric data type 
        not_numeric_columns, not_date_columns, not_string_columns = self.dqc.types_check()

        for column_name in not_numeric_columns:
            self.df[column_name] = self.df[column_name].apply(np.int64)

        # conversion date columns with not date data type         
        for column_name in not_date_columns:
            self.df[column_name] = pd.to_datetime(self.df[column_name], dayfirst=dayfirst)

        # conversion string columns with not string data type         
        for column_name in not_string_columns:
            self.df[column_name] = self.df[column_name].apply(str)

        # converting columns of different data types to the columns of the same data type (data validity)
        for column in self.df.columns:
            if self.df[column].dtype == "object": # we check columns with an object type, which could potentially be a column with data of different types
                expected_type = type(self.df[column].iloc[0]) # take the value type of the first row
                self.df = self.df[column].apply(expected_type) # try to convert column samples with a wrong type into the right type (expected_type)

        print("\nDataFrame column types processed.")
        return self.df
    
    def ts_nonstatinarity_processing(self, column: str) -> pd.DataFrame:
        """
        Processes a specified column in the DataFrame to achieve stationarity.

        This method applies differencing to a specified column in the DataFrame
        until the column achieves stationarity, as determined by a stationarity check
        method (`ts_stationarity_check`). After each differencing operation, the mean
        of the differenced array is appended to maintain the array size. The process
        repeats until the column is considered stationary.

        Parameters:
        - column (str): The name of the column to process for stationarity.

        Returns:
        - pd.DataFrame: The DataFrame with the processed column now potentially stationary.

        Note:
        - The method assumes the existence of a `ts_stationarity_check` method within an `eda`
        attribute of the class, which should return a tuple where the first element is a string
        indicating if the column is "stationary" or not.
        - This method directly modifies the original DataFrame stored in `self.df` and also returns
        it for convenience.
        - It is important to note that differencing can make the data stationary by removing trends
        and seasonality, but the process may also remove some information from the data.

        Example:
        ```
        # Assuming `self` is an instance of a class containing the DataFrame `df` and an EDA tool `eda`.
        processed_df = self.ts_nonstationarity_processing('my_column')
        print(processed_df)
        ```

        After execution, the specified column will have been processed to remove non-stationarity,
        and the method prints a confirmation message.
        """
        diff_column: np.ndarray = np.diff(self.df[column])
        diff_column: np.ndarray  = np.append(diff_column, np.mean(diff_column))
        self.df[column] = diff_column

        while self.eda.ts_stationarity_check(column)[0] != "stationary":
            diff_column = self.df[column]
            diff_column = np.diff(diff_column)
            diff_column = np.append(diff_column, np.mean(diff_column))
            self.df[column] = diff_column

        print(f"Non-stationarity for {column} column processed.")
        return self.df

    def transform(self, outliers_columns: List[str], inconsistency_columns: List[str], dayfirst: bool=True) -> pd.DataFrame:
        """
        Perform a series of data transformation steps.

        This method executes a series of data transformation steps on the DataFrame:
        1. Process outliers in specified columns.
        2. Remove rows with missing values.
        3. Process inconsistent or duplicated rows based on specified columns.
        4. Convert DataFrame columns to appropriate data types.

        Parameters:
        -----------
        outliers_columns : List[str]
            A list of column names in which to process outliers.
        inconsistency_columns : List[str]
            A list of column names used to check for inconsistency or duplication.
        dayfirst : bool, optional (default=True)
            Whether to interpret dates with the day first (e.g., '10/01/2022' as January 10, 2022).

        Returns:
        --------
        pd.DataFrame
            The transformed DataFrame after all the specified data transformation steps.

        Notes:
        ------
        Each transformation step is applied sequentially to the DataFrame.
        The resulting DataFrame reflects all the transformations performed.
        """
        transformations: list = [self.outliers_processing(outliers_columns), self.na_processing(), self.inconst_dupl_processing(inconsistency_columns), self.data_type_conversion(dayfirst=dayfirst)]
        for transformation in transformations:
            transformation

        return self.df