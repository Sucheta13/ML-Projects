# Student Performance Prediction – End-to-End ML Pipeline

A modular machine-learning pipeline that predicts students' math scores from demographic and academic features (gender, parental education, lunch type, test-preparation course, reading and writing scores).

## What it does

- **Data ingestion** – reads the raw CSV, saves it, and creates an 80/20 train/test split.
- **Data transformation** – imputation, scaling and one-hot encoding with scikit-learn pipelines; the fitted preprocessor is saved to `artifacts/preprocessor.pkl`.
- **Model training** – compares Linear Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost and CatBoost with hyperparameter search, and saves the best model (R² ≥ 0.6) to `artifacts/model.pkl`.
- Custom exception handling and logging.

## Project structure

```
Notebook/          EDA and model-training notebooks + dataset
src/components/    data_ingestion.py, data_transformation.py, model_trainer.py
src/               utils.py, logger.py, exception.py
artifacts/         generated train/test data and saved model
```

## Run it

```bash
pip install -r requirements.txt
python -m src.components.data_ingestion
```

## Tech

Python, pandas, NumPy, scikit-learn, CatBoost, XGBoost, seaborn, matplotlib
