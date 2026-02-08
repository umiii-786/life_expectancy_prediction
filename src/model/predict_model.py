from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd
from src.logging_config import logger
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from dvclive import Live


def load_dataset(path: str) -> tuple:
    try:
        train_file_path = os.path.join(path, 'selected_feature_train.csv')
        test_file_path = os.path.join(path, 'selected_feature_test.csv')

        if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
            logger.error("Training or Testing dataset not found")
            raise 
        # FileNotFoundError("Training or Testing dataset not found")

        train_ds = pd.read_csv(train_file_path)
        test_ds = pd.read_csv(test_file_path)

        if train_ds.empty or test_ds.empty:
            logger.error("Training or testing dataset is empty")
            raise 

        logger.debug("Training and Testing datasets loaded successfully")
        return train_ds, test_ds

    except pd.errors.EmptyDataError:
        logger.error("Training or testing CSV file is empty")
        raise
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise


def load_pipeline(path: str) -> Pipeline:
    try:
        if not os.path.exists(path):
            logger.error("Saved model pipeline not found")
            raise 
        with open(path, 'rb') as f:
            pipe = pickle.load(f)

        logger.debug("Model pipeline loaded successfully")
        return pipe

    except pickle.UnpicklingError:
        logger.error("Failed to unpickle model pipeline")
        raise
    except Exception as e:
        logger.error(f"Error loading model pipeline: {e}")
        raise


def getPrediction(
    train_ds: pd.DataFrame,
    test_ds: pd.DataFrame,
    pipe: Pipeline
) -> None:
    try:
        target_col = 'Life expectancy '

        if target_col not in train_ds.columns or target_col not in test_ds.columns:
            logger.error("Target column 'Life expectancy ' not found")
            raise 

        X_train = train_ds.drop(target_col, axis=1)
        y_train = train_ds[target_col]

        X_test = test_ds.drop(target_col, axis=1)
        y_test = test_ds[target_col]

        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        with Live(save_dvc_exp=True) as live:
            live.log_metric(
                'r2_score_train',
                r2_score(y_true=y_train, y_pred=y_pred_train)
            )
            live.log_metric(
                'rmse_train',
                np.sqrt(mean_squared_error(y_true=y_train, y_pred=y_pred_train))
            )
            live.log_metric(
                'r2_score_test',
                r2_score(y_true=y_test, y_pred=y_pred_test)
            )
            live.log_metric(
                'rmse_test',
                np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred_test))
            )

        logger.info("Model evaluation metrics logged successfully")

    except ValueError as e:
        logger.error(f"Prediction value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise


def main() -> None:
    try:
        path = os.path.join(
            'data', 'feature_engineering', 'feature_selection'
        )

        train_ds, test_ds = load_dataset(path)
        pipe = load_pipeline('models/rf.pkl')
        getPrediction(train_ds, test_ds, pipe)

        logger.info("✅ Model evaluation pipeline completed successfully")

    except Exception as e:
        logger.critical(f"❌ Evaluation pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
