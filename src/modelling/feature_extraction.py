from ..data.etl_layer import ETL

from sklearn.model_selection import train_test_split



class Feature_Extraction:
    def __init__(self):
        pass

    def get_fe_processed_df(self, processed_sales_df, full_processed_monthly_df):
        processed_sales_df["month"] = processed_sales_df["date"].dt.to_period("M")
        
        month_mean_price = processed_sales_df.groupby(["month", "shop_id", "item_id"])["item_price"].mean()
        month_mean_price.reset_index(drop=True, inplace=True)

        full_processed_monthly_df["mean_month_price"] = month_mean_price

        full_processed_monthly_df['month_num'] = full_processed_monthly_df['month'].dt.month

        lags = [i for i in range(1, 13)]

        for lag in lags:
            full_processed_monthly_df[f"item_cnt_month_lag_{lag}"] =  full_processed_monthly_df['item_cnt_month'].shift(lag)

        monthly_etl = ETL(df=full_processed_monthly_df)
        monthly_etl.transform(["item_cnt_month"], ["item_cnt_month"])

        full_processed_monthly_df = monthly_etl.get_data()

        full_processed_monthly_df.drop("month", axis=1, inplace=True)

        return full_processed_monthly_df
    
 