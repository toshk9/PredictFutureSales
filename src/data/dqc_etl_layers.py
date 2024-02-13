import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
import sqlite3
from typing import List, Tuple
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import kpss, adfuller 
import statsmodels.api as sm
import statsmodels

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
        Processes outliers in specified columns by replacing them with the column mode.

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

        self.df.to_csv('/data/processed/' + file_name + '.csv')
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
        with sqlite3.connect(f"data/processed/{file_name}.db") as db: 
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
            self.df.loc[outliers_check[column], column] = self.df[column].mode()
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
            if self.df.dtypes[column] == object: # we check columns with an object type, which could potentially be a column with data of different types
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
        print(f"\nDataset {self.df_title} Info:\n")
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
                
        print(f"\nСolumns with presumably the wrong data type: \nnot numeric type columns: {not_numeric_columns},\nnot date type columns: {not_date_columns},\nnot string type columns: {not_string_columns}")
        return not_numeric_columns, not_date_columns, not_string_columns

class EDA:
    """
    Exploratory Data Analysis (EDA) class for performing various statistical checks and visualizations on a pandas DataFrame.

    Attributes:
    - df (pd.DataFrame): A pandas DataFrame containing the data for analysis.

    Methods:
    - features_corr_check: Checks for correlations between specified features using various methods and visualizes them using a heatmap.
    - normal_distr_check: Checks for normal distribution in specified features through histograms and Q-Q plots.
    - ts_autocorr_check: Checks for autocorrelation in a time series column up to a specified number of lags and identifies significant lags.
    - ts_stationarity_check: Checks the stationarity of a time series column using both the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.
    - heterosked_check: Performs a Breusch-Pagan test to check for heteroskedasticity in the residuals of a linear model.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the EDA class with a DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to perform EDA on.
        """
        self.df: pd.DataFrame = df

    def features_corr_check(self, columns: List[str], methods: list=["pearson"], **kwargs) -> None:
        """
        Checks for and visualizes the correlation between specified columns in the DataFrame using specified methods.

        This method calculates the correlation between the specified columns using one or more correlation methods 
        (e.g., Pearson, Kendall, Spearman) and visualizes the correlation matrix using a heatmap. This can be useful 
        for identifying potential relationships between variables that may warrant further investigation.

        Parameters:
        - columns (List[str]): A list of column names within the DataFrame for which correlations are to be calculated.
        - methods (list, optional): A list of strings indicating the correlation methods to use. Default is ["pearson"].
                                Other valid options include "kendall" and "spearman".
        - **kwargs: Additional keyword arguments to be passed to the pandas `corr` method.

        Returns:
        - None: This method does not return a value but displays a heatmap visualization of the correlation matrix for 
                each specified method.

        Example:
        ```python
        eda = EDA(my_dataframe)
        eda.features_corr_check(columns=['feature1', 'feature2', 'feature3'], methods=['pearson', 'spearman'])
        ```
        This will calculate and display the correlation matrix for 'feature1', 'feature2', and 'feature3' using both the
        Pearson and Spearman correlation coefficients.
        """
        for method in methods:
            sales_train_corr = self.df[columns].corr(method=method, **kwargs)
            sns.heatmap(sales_train_corr, annot=True)
            plt.title(f"Checking for correlation using {method} method")
            plt.show()

    def normal_distr_check(self, columns: List[str], **kwargs) -> None:
        """
        Visualizes the distribution of specified columns in the DataFrame and checks for normality.

        This method generates histograms and Q-Q (quantile-quantile) plots for each specified column to help assess
        whether the data distributions approximate a normal distribution. Histograms provide a visual representation
        of the data distribution, while Q-Q plots compare the quantiles of the data to the quantiles of a standard
        normal distribution. Deviations from the line in a Q-Q plot indicate departures from normality.

        Parameters:
        - columns (List[str]): A list of column names within the DataFrame for which distributions are to be visualized.
        - **kwargs: Additional keyword arguments to be passed to the matplotlib `hist` function for histogram customization.

        Returns:
        - None: This method does not return a value but displays histograms and Q-Q plots for each specified column.

        Example:
        ```python
        eda = EDA(my_dataframe)
        eda.normal_distr_check(columns=['feature1', 'feature2'])
        ```
        This will generate histograms and Q-Q plots for 'feature1' and 'feature2', allowing for a visual assessment
        of their adherence to a normal distribution.
        """
        _, ax = plt.subplots(len(columns), 2)
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        for i, column in enumerate(columns):
            ax[i][0].hist(self.df[column], **kwargs)
            ax[i][0].set_title(f"{column} histogram")
            ax[i][0].set_xlabel(column)

            stats.probplot(self.df[column], plot=ax[i][1])
        plt.show()
    
    def ts_autocorr_check(self, column, lags : int, significance_level: float=0.05, **kwargs) -> list[int]:
        """
        Performs the Ljung-Box test for autocorrelation on a specified column of the DataFrame and returns the lags
        where autocorrelation is statistically significant.

        The Ljung-Box test is used to determine if there are any statistically significant autocorrelations at lag values
        up to the specified number. This method is particularly useful in time series analysis to identify if the
        observed time series is random or if there are underlying patterns in the autocorrelation function.

        Parameters:
        - column (str): The name of the column in the DataFrame to perform the autocorrelation check on.
        - lags (int): The maximum number of lags to test for autocorrelation.
        - significance_level (float, optional): The significance level to determine if the autocorrelation at a specific lag is statistically significant. Defaults to 0.05.
        - **kwargs: Additional keyword arguments to be passed to the `acorr_ljungbox` function.

        Returns:
        - list[int]: A list of lag values where the autocorrelation is found to be statistically significant according to the Ljung-Box test.

        Example:
        ```python
        eda = EDA(my_dataframe)
        significant_lags = eda.ts_autocorr_check(column='my_time_series_column', lags=20)
        print(f"Significant lags up to 20: {significant_lags}")
        ```
        This example checks for significant autocorrelation up to 20 lags in the 'my_time_series_column' and prints the lags where significant autocorrelation is detected.
        """
        acorr_stats: pd.DataFrame = acorr_ljungbox(self.df[column], lags=list(range(lags+1)), **kwargs)
        acorr_lags: list[int] = []
        for lag, p_value in enumerate(acorr_stats["lb_pvalue"][1:]):
            if p_value <= significance_level:
                acorr_lags.append(lag)
        return acorr_lags
    
    def ts_stationarity_check(self, column: str, significance_level: float=0.05, adf_params: dict={}, kpss_params: dict={}) -> tuple:
        """
        Performs stationarity checks on a specified time series column using Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.

        The ADF test evaluates the null hypothesis that a unit root is present in a time series, indicating that the series is non-stationary. 
        Conversely, the KPSS test evaluates the null hypothesis that the series is trend stationary, meaning it exhibits stationarity around a deterministic trend.

        Parameters:
        - column (str): The name of the column in the DataFrame representing the time series to be checked for stationarity.
        - significance_level (float, optional): The significance level used to determine the rejection of the null hypothesis. Defaults to 0.05.
        - adf_params (dict, optional): Additional parameters to be passed to the `adfuller` function. Defaults to an empty dictionary.
        - kpss_params (dict, optional): Additional parameters to be passed to the `kpss` function. Defaults to an empty dictionary.

        Returns:
        - tuple: A tuple containing:
            - stationarity (str): The classification of the time series stationarity. Possible values are 'stationary', 'non-stationary', 'trend stationary', 'difference stationary', or 'not defined'.
            - adf_stats (tuple): A tuple containing the ADF test statistics, including the test statistic, p-value, and critical values.
            - kpss_stats (tuple): A tuple containing the KPSS test statistics, including the test statistic, p-value, and critical values.

        Example:
        ```python
        eda = EDA(my_dataframe)
        stationarity, adf_stats, kpss_stats = eda.ts_stationarity_check(column='my_time_series_column', significance_level=0.05)
        print(f"Stationarity: {stationarity}")
        print(f"ADF test statistics: {adf_stats}")
        print(f"KPSS test statistics: {kpss_stats}")
        ```
        This example performs stationarity checks on the 'my_time_series_column' and prints the stationarity classification as well as the ADF and KPSS test statistics.
        """
        stationarity: str = "not defined"

        # ADF
        # H0: Series is non-stationary, or series has a unit root.
        # H1: Series is stationary, or series has no unit root. 
        try:
            adf_stats: tuple = adfuller(self.df[column], **adf_params)
        except ValueError:
            print("Invalid input, x is constant")
            stationarity = "stationary"
            return (stationarity, "x is constant", "x is constant") 
        
        # KPSS
        # H0: Series is trend stationary or series has no unit root.
        # H1: Series is non-stationary, or series has a unit root.
        try:
            kpss_stats: tuple = kpss(self.df[column], **kpss_params)
        except OverflowError:
            print("OverflowError: cannot convert float infinity to integer.")
            return (stationarity, adf_stats, "only adf stats") 
        
        significance_level_perc: str = str(int(significance_level * 100)) + "%"

        try:
            if (kpss_stats[0] < kpss_stats[3][significance_level_perc]) and (adf_stats[0] < adf_stats[4][significance_level_perc]) and \
            (kpss_stats[1] > significance_level) and (adf_stats[1] < significance_level):
                stationarity = "stationary"
            elif (kpss_stats[0] > kpss_stats[3][significance_level_perc]) and (adf_stats[0] > adf_stats[4][significance_level_perc]) and \
            (kpss_stats[1] < significance_level) and (adf_stats[1] > significance_level):
                stationarity = "non-stationary"
            elif (kpss_stats[0] < kpss_stats[3][significance_level_perc]) and (adf_stats[0] > adf_stats[4][significance_level_perc]) and \
            (kpss_stats[1] > significance_level) and (adf_stats[1] > significance_level):
                stationarity = "trend stationary"
            elif (kpss_stats[0] > kpss_stats[3][significance_level_perc]) and (adf_stats[0] < adf_stats[4][significance_level_perc]) and \
            (kpss_stats[1] < significance_level) and (adf_stats[1] < significance_level):
                stationarity = "difference stationary"
            else:
                print("The stationarity of the series cannot be verified.")

        except KeyError:
            if (kpss_stats[1] > significance_level) and (adf_stats[1] < significance_level):
                stationarity = "stationary"
            elif (kpss_stats[1] < significance_level) and (adf_stats[1] > significance_level):
                stationarity = "non-stationary"
            elif (kpss_stats[1] > significance_level) and (adf_stats[1] > significance_level):
                stationarity = "trend stationary"
            elif (kpss_stats[1] < significance_level) and (adf_stats[1] < significance_level):
                stationarity = "difference stationary"
            else:
                print("The stationarity of the series cannot be verified.")

        except Exception:
            print("The stationarity of the series cannot be verified with the given parameters.")

        return (stationarity, adf_stats, kpss_stats)
    
    def heterosked_check(self, X_columns: List[str], y_column: str, significance_level: float=0.05) -> tuple:
        """
        Performs heteroskedasticity test on the residuals of a linear regression model.

        The Breusch-Pagan test is employed to determine whether the variance of the errors from a regression model is constant (homoskedastic) or varies with the independent variables (heteroskedastic).

        Parameters:
        - X_columns (List[str]): The list of column names representing the independent variables.
        - y_column (str): The name of the column representing the dependent variable.
        - significance_level (float, optional): The significance level used to determine the rejection of the null hypothesis. Defaults to 0.05.

        Returns:
        - tuple: A tuple containing:
            - heteroskedasticity (bool): A boolean indicating whether heteroskedasticity is present (True) or not (False).
            - bp_statistics (dict): A dictionary containing the Breusch-Pagan test statistics, including the LM statistic, LM p-value, F-statistic, and F p-value.

        Example:
        ```python
        eda = EDA(my_dataframe)
        heteroskedasticity, bp_stats = eda.heterosked_check(X_columns=['feature1', 'feature2'], y_column='target', significance_level=0.05)
        if heteroskedasticity:
            print("Heteroskedasticity is present.")
        else:
            print("No evidence of heteroskedasticity.")
        print(f"Breusch-Pagan test statistics: {bp_stats}")
        ```
        This example performs heteroskedasticity test on the residuals of a linear regression model with 'feature1' and 'feature2' as independent variables and 'target' as the dependent variable. It then prints the result and the Breusch-Pagan test statistics.
        """
        y: pd.Series = self.df[y_column]  
        X: pd.DataFrame = self.df[X_columns] 
        X: np.ndarray = np.column_stack((np.ones(len(y)), X))

        model: statsmodels.regression.linear_model.RegressionResultsWrapper = sm.OLS(y, X).fit()

        predictions: pd.Series = model.predict(X)
        
        residuals: pd.Series = y - predictions

        bp_test: tuple = het_breuschpagan(resid=residuals, exog_het=X)

        labels: List[str] = ['LM statistic', 'LM p-value', 'F-statistic', 'F p-value']
        bp_statistics: dict = dict(zip(labels, bp_test))
        
        heteroskedasticity: bool = False
        if (bp_statistics["LM p-value"] < significance_level) and (bp_statistics["F p-value"] < significance_level):
            heteroskedasticity = True
        
        return (heteroskedasticity, bp_statistics)
            

