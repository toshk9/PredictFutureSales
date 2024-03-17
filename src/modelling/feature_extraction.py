import pandas as pd


class Feature_Extraction:
    """
    A class for feature extraction from daily and monthly datasets.

    Args:
        daily_df (pd.DataFrame): Daily dataset containing transactional data.
        monthly_df (pd.DataFrame, optional): Monthly dataset containing aggregated data. Defaults to None.

    Attributes:
        daily_df (pd.DataFrame): Daily dataset containing transactional data.
        monthly_df (pd.DataFrame): Monthly dataset containing aggregated data.

    Methods:
        get_fe_df(): Combines daily and monthly datasets and computes additional features.
        create_monthly_df(): Aggregates daily data to create a monthly dataset.
    """
    def __init__(self, daily_df: pd.DataFrame, monthly_df: pd.DataFrame=None) -> None:
        """
        Initializes a Feature_Extraction object.

        Args:
            daily_df (pd.DataFrame): Daily dataset containing transactional data.
            monthly_df (pd.DataFrame, optional): Monthly dataset containing aggregated data. Defaults to None.
        """
        self.daily_df: pd.DataFrame = daily_df

        if self.daily_df.dtypes["date"] == "object":
            self.daily_df["date"] = pd.to_datetime(self.daily_df["date"], dayfirst=True)

        self.daily_df["month"] = self.daily_df["date"].dt.to_period("M")

        self.monthly_df: pd.DataFrame = self.create_monthly_df() if monthly_df is None else monthly_df

    def get_fe_df(self) -> pd.DataFrame:
        """
        Combines daily and monthly datasets and computes additional features.

        Returns:
            pd.DataFrame: Feature-engineered dataset.
        """
        month_mean_price: pd.Series = self.daily_df.groupby(["month", "shop_id", "item_id"])["item_price"].mean()
        month_mean_price.reset_index(drop=True, inplace=True)
        
        fe_df: pd.DataFrame = self.monthly_df
        fe_df["mean_month_price"] = month_mean_price

        fe_df['month_num'] = fe_df['month'].dt.month

        fe_df.drop("month", axis=1, inplace=True)

        return fe_df
    
    def create_monthly_df(self) -> pd.DataFrame:
        """
        Aggregates daily data to create a monthly dataset.

        Returns:
            pd.DataFrame: Monthly dataset containing aggregated data.
        """
        monthly_sales: pd.DataFrame = self.daily_df.groupby(["month", "shop_id", "item_id"])["item_cnt_day"].sum() 
        monthly_sales = pd.DataFrame(monthly_sales)
        monthly_sales = monthly_sales.rename(columns={'item_cnt_day': 'item_cnt_month'})
        monthly_sales = monthly_sales.reset_index()
        return monthly_sales
    
 