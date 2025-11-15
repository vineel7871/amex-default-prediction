# American Express Default Prediction

A machine learning project using PySpark to predict credit card default risk using American Express transaction data.

## Overview

This project implements a binary classification model to predict whether a customer will default on their credit card payments. The analysis uses distributed computing with Apache Spark to handle large-scale transaction data efficiently.

## Features

- **Data Processing**: Automated data loading and preprocessing pipeline
- **Feature Engineering**: 
  - Categorical variable encoding using StringIndexer
  - Missing value imputation with median strategy
  - Feature selection based on missing data thresholds
- **Model Training**: Random Forest Classifier implementation
- **Evaluation Metrics**: AUC, Precision, Recall, and F1-Score

## Dataset

The project uses two main data files:
- `train_data.csv`: Customer transaction features
- `train_labels.csv`: Binary target labels (default/no default)

## Requirements

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install pyspark
```

## Usage

Open and run the `pyspark.ipynb` notebook to:

1. Initialize a Spark session
2. Load and explore the data
3. Preprocess features (handle missing values, encode categorical variables)
4. Train the Random Forest model
5. Evaluate model performance

## Project Structure

```
.
├── data/
│   ├── train_data.csv
│   └── train_labels.csv
├── pyspark.ipynb
├── README.md
└── pyproject.toml
```

## Model Performance

The Random Forest classifier achieves:
- Training on 80% of the data
- Evaluation on 20% test split
- Metrics: AUC, Precision, Recall, F1-Score

## Preprocessing Steps

1. **Data Loading**: Load CSV files with schema inference
2. **Column Classification**: Separate numerical, categorical, date, and ID columns
3. **Missing Value Analysis**: Identify columns with high missing rates
4. **Feature Selection**: Remove columns with >50% missing values
5. **Imputation**: Fill remaining missing values with median
6. **Encoding**: Convert categorical variables to numeric indices
7. **Vector Assembly**: Combine all features into a single vector for ML

## License

MIT

## Acknowledgments

- Dataset: American Express Default Prediction Challenge
- Framework: Apache Spark MLlib
