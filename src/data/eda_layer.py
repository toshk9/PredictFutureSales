import pandas as pd
import numpy as np

from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import kpss, adfuller 
from scipy import stats


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
            df_corr: pd.DataFrame = self.df[columns].corr(method=method, **kwargs)
            sns.heatmap(df_corr, annot=True)
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