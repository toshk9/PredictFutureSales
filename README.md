# PredictFutureSales

## Overview
This project is dedicated to solving the Predict Future Sales competition task on Kaggle. The project implements various data processing, analysis and modelling techniques for the Predict Future Sales competition dataset. The project is organized into different layers, including ETL, DQC, EDA and Explainability, Feature Extraction, Moder Error Analysis, etc., to effectively handle and analyze the dataset. 

## Directory Structure
project_directory/
│
├── data/
│ ├── data_validation/
│ │ ├── 
│ │ ├── schema_1.pbtxt
│ │ ├── schema_2.pbtxt
│ │ └── ...
│ │ └── stats_1.pb
│ │ ├── stats_2.pb
│ │ └── ...
│ ├── model_predictions/
│ │ ├── predictions_run_1.csv
│ │ ├── predictions_run_2.csv
│ │ └── ...
│ ├── processed/
│ │ ├── processed_data_1.csv
│ │ ├── processed_data_2.csv
│ │ └── ...
│ └── raw/
│ ├── raw_data_1.csv
│ ├── raw_data_2.csv
│ └── ...
│
├── models/
│ ├── CatBoost
│ ├── XGBoost.bin
│ ├── Linear Regression.pkl
│ └── ...
│
├── notebooks/
│ ├── data_analysis.ipynb
│ └── modelling.ipynb
│
└── src/
├── data/
│ ├── dqc_layer.py
│ ├── eda_layer.py
│ └── etl_layer.py
│
├── modelling/
│ ├── data_validation.py
│ ├── explainability_layer.py
│ ├── feature_extraction.py
│ ├── feature_importance.py
│ ├── hyperparameter_optimization.py
│ └── model_error_analysis.py
│
├── init.py
└──  main.py
├── data_models_download.py
└── requirements.txt

## Installation
To run the project locally, follow these steps:
1. Clone the repository: `git clone <repository-url>`
2. Install the required Python dependencies: `pip install -r requirements.txt`
3. Ensure that the data files are placed in the `data/` directory and the models files are placed in the `models/` directory.

## Usage
1. **Data Processing**: The `src/data` directory contains modules and scripts for data processing:
   - `dqc_layer.py`: This module contains a class with methods for the Data Quality Check (DQC) process. It includes functionalities to identify and handle missing values, outliers, and inconsistencies in the data.
   - `eda_layer.py`: This module provides a class with methods for Exploratory Data Analysis (EDA). It includes functions for visualizing data distributions, correlations, and relationships between features.
   - `etl_layer.py`: This module offers a class with methods for the Extract, Transform, Load (ETL) process. It includes functions to preprocess, clean, and transform raw data into a format suitable for model training.

2. **Modeling**: The `src/modelling` directory contains modules and scripts for model development and analysis:
   - `data_validation.py`: This module contains a class with methods for Data Validation using TensorFlow Data Validation. It includes functionalities to validate and monitor data quality, schema evolution, and data drift.
   - `explainability_layer.py`: This module provides a class with methods for model explainability using SHAP (SHapley Additive exPlanations). It includes functionalities to interpret and explain model predictions at the feature level.
   - `feature_extraction.py`: This module offers a class with methods for Feature Extraction. It includes functions to extract relevant features from raw data, such as text, images, or time series.
   - `feature_importance.py`: This module provides a class with methods for measuring Feature Importance using techniques like Boruta and Impurity-based methods. It includes functionalities to identify the most influential features for model performance.
   - `hyperparameter_optimization.py`: This module contains a class with methods for Hyperparameter Optimization using Hyperopt. It includes functionalities to tune model hyperparameters automatically and efficiently.
   - `model_error_analysis.py`: This module provides a class with methods for Model Error Analysis. It includes functionalities to analyze prediction errors, identify patterns of error, and assess model performance for different subsets of data.

3. **Inference**: Use `src/main.py` to run model inference from the command line. Execute the following command:
python3 src/main.py --inference ../path/to/inference_data.csv

Replace `../path/to/inference_data.csv` with the path to the data for inference.

4. **Data and Model Download**: Use `data_models_download.py` to download remotely stored data and trained models from Google Drive to your local storage.

## Contributing
Contributions are welcome! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Requirements
The required Python packages are listed in [requirements.txt](requirements.txt). You can install them using pip:
pip install -r src/requirements.txt

## Acknowledgements
- The project makes use of various open-source libraries and frameworks.
- Inspiration for this project was drawn from real-world data analysis scenarios.