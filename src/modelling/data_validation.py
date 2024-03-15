import tensorflow_data_validation as tfdv
import tensorflow_metadata

from typing import Union

import pandas as pd

import os

class Data_Validation:
    def __init__(self, stats: tensorflow_metadata.proto.v0.statistics_pb2.DatasetFeatureStatisticsList=None, schema:tensorflow_metadata.proto.v0.schema_pb2.Schema=None) -> None:
        self.stats = stats
        self.schema = schema
            
    def get_stats(self) -> tensorflow_metadata.proto.v0.statistics_pb2.DatasetFeatureStatisticsList:
        if self.stats == None:
            raise "There are no statistics, use 'create_stats()' to create."
        else:
            return self.stats
    
    def get_schema(self) -> tensorflow_metadata.proto.v0.schema_pb2.Schema:
        if self.schema == None:
            self.create_schema()
        return self.schema
    
    def create_stats(self, ds: Union[pd.DataFrame, str]) -> None:
        if type(ds) is str:
            file_path: str = ds
            file_extension: str = os.path.splitext(file_path)[1]

            if file_extension == '.csv':
                self.stats: tensorflow_metadata.proto.v0.statistics_pb2.DatasetFeatureStatisticsList = tfdv.generate_statistics_from_csv(file_path)
            elif file_extension == '.tfrecord':
                self.stats: tensorflow_metadata.proto.v0.statistics_pb2.DatasetFeatureStatisticsList = tfdv.generate_statistics_from_tfrecord(file_path)
            else:
                raise "Wrong data format."
        elif type(ds) is pd.DataFrame:
            self.stats: tensorflow_metadata.proto.v0.statistics_pb2.DatasetFeatureStatisticsList = tfdv.generate_statistics_from_dataframe(ds)
        else:
            raise "Wrong data type in 'ds' argument."
        
        print("The Data Stats has been successfully created")

    def create_schema(self) -> None:
        print("There is no schema.")
        print("Trying to create a schema...")
        if self.stats != None:
            self.schema: tensorflow_metadata.proto.v0.schema_pb2.Schema = tfdv.infer_schema(statistics=self.stats)
            print("The schema has been successfully created")
            tfdv.display_schema(self.schema)
        else:
            raise "There are no statistics, use 'create_stats()' to create."

    def save_schema(self, schema_title: str) -> None:
        if self.schema == None:
            self.create_schema()
        
        tfdv.write_schema_text(self.schema, f"../data/data_schemas/{schema_title}.pbtxt")

        print(f"The data schema has been successfully saved in ../data/data_schemas/{schema_title}.pbtxt")

    def load_schema(self, schema_path: str) -> None:
        self.schema: tensorflow_metadata.proto.v0.schema_pb2.Schema = tfdv.load_schema_text(schema_path)

    def validate_data(self, new_stats: tensorflow_metadata.proto.v0.statistics_pb2.DatasetFeatureStatisticsList, drift_feature_threshold_dict: dict=None, skew_feature_threshold_dict: dict=None) -> None:
        if self.stats == None:
            raise "There are no statistics, use 'create_stats()' to create."
        elif self.schema == None:
            self.create_schema()

        if drift_feature_threshold_dict is not None:
            for feature, threshold in drift_feature_threshold_dict.items():
                tfdv.get_feature(self.schema, feature).drift_comparator.infinity_norm.threshold = threshold
        
        if skew_feature_threshold_dict is not None:
            for feature, threshold in skew_feature_threshold_dict.items():
                tfdv.get_feature(self.schema, feature).skew_comparator.infinity_norm.threshold = threshold

        anomalies = tfdv.validate_statistics(statistics=new_stats, schema=self.schema, serving_statistics=self.stats, previous_statistics=self.stats)

        tfdv.display_anomalies(anomalies)

        tfdv.visualize_statistics(lhs_statistics=self.stats, rhs_statistics=new_stats, lhs_name="Serving Data Stats", rhs_name="New Data Stats")
