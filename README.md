# PredictFutureSales

## Overview
This project is designed to perform an end-to-end data analysis pipeline, including Extraction, Transformation, Loading (ETL), Data Quality Control (DQC), and Exploratory Data Analysis (EDA) processes. The pipeline is implemented using Python and leverages various libraries such as pandas, numpy, matplotlib, seaborn, and statsmodels.

## Project Structure
The project directory is structured as follows:
- **data/**
  - **raw/**: Contains the raw CSV files with the initial dataset.
  - **processed/**: Directory for storing processed datasets.
- **notebooks/**: Holds Jupyter notebooks for data analysis, preprocessing, and visualization.
  - **data_analysis.ipynb**: Jupyter notebook where data overview, preprocessing, and visualization are performed using ETL, DQC, and EDA processes.
- **src/**
  - **etl_layer.py**: Python module for performing Extraction, Transformation, and Loading operations on datasets.
  - **dqc_layer.py**: Python module for Data Quality Control, including checks for missing values, outliers, data consistency, and types.
  - **eda_layer.py**: Python module for Exploratory Data Analysis, covering statistical checks and visualizations.
- **config.py**: Configuration file containing paths to raw data files.


## Installation
To run the project locally, follow these steps:
1. Clone the repository: `git clone <repository-url>`
2. Install the required Python dependencies: `pip install -r requirements.txt`
3. Ensure that the raw CSV files are placed in the `data/raw/` directory.

## Usage
1. Update the `config.py` file with the paths to your raw CSV files.
2. Use the provided Jupyter notebook `data_analysis.ipynb` to perform data analysis.
3. Utilize the functionalities provided in the `etl_layer.py`, `dqc_layer.py`, and `eda_layer.py` modules for custom data processing and analysis.

## Contributing
Contributions are welcome! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
- The project makes use of various open-source libraries and frameworks.
- Inspiration for this project was drawn from real-world data analysis scenarios.
