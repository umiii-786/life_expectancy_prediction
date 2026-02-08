from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import os
import pandas as pd
import yaml
import pickle
from dvclive import Live
from src.logging_config import logger


def load_YML(path: str) -> dict:
    try:
        if not os.path.exists(path):
            logger.error(f"YAML file not found: {path}")
            raise 

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        logger.debug(f"Loaded YAML file Successfully")
        return config['rf_parameter']

    except KeyError:
        logger.debug("Missing 'rf_parameter' section in params.yaml")
        raise 
    except yaml.YAMLError as e:
        logger.debug(f"Invalid YAML format: {e}")
        raise 
    except Exception as e:
        logger.debug(f"Error loading YAML file: {e}")
        raise 


def load_dataset(path: str) -> pd.DataFrame:
    try:
        file_path = os.path.join(path, 'selected_feature_train.csv')

        if not os.path.exists(file_path):
            logger.error("Training dataset not found")
            raise 

        train_ds = pd.read_csv(file_path)

        if train_ds.empty:
            logger.error("Training dataset is empty")
            raise 
        logger.debug('Data set Loaded Successfully')
        return train_ds

    except pd.errors.EmptyDataError:
        logger.error("Training CSV file is empty")
        raise 
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise 


def Train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict
):
    try:
        if X_train.empty or y_train.empty:
            logger.error("Training features or target is empty")
            raise 

        required_params = ['max_depth', 'n_estimators']
        for param in required_params:
            if param not in parameters:
                logger.error(f"Missing parameter '{param}' in YAML")
                raise 

        trf = ColumnTransformer(
            transformers=[
                (
                    "one_hot",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    [1, 15]
                ),
                (
                    "scaling",
                    StandardScaler(),
                    [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -2, -1]
                ),
            ],
            remainder="passthrough"
        )
        logger.debug('Column Transformer Builded')

        pipe = Pipeline(steps=[
            ('preprocessing', trf),
            ('model', GradientBoostingRegressor(max_depth=parameters['max_depth'],
                                                max_features=0.5,
                                                n_estimators=parameters['n_estimators'],
                                                subsample=0.5))
        ])

        logger.debug('Python pipeline Builded Transformer Builded')
        pipe.fit(X_train, y_train)
        logger.debug('Successfully Trainned Model')
        return pipe

    except ValueError as e:
        logger.error(f"Model training input error: {e}")
        raise 
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise 


def save_pipeline(pipe) -> None:
    try:
        if pipe is None:
            logger.error("Pipeline object is None")
            raise 

        os.makedirs('models', exist_ok=True)

        with open('models/rf.pkl', 'wb') as f:
            pickle.dump(pipe, f)
        logger.debug('Pipeline Saved Successfully')
    except PermissionError:
        logger.error("Permission denied while saving model")
        raise 
    except Exception as e:
        logger.error(f"Error saving model pipeline: {e}")
        raise 


def main() -> None:
    try:
        path = os.path.join(
            'data', 'feature_engineering', 'feature_selection'
        )

        train_ds = load_dataset(path)
        parameters = load_YML('params.yaml')

        if 'Life expectancy ' not in train_ds.columns:
            logger.error("Target column 'Life expectancy ' not found")
            raise 

        X_train = train_ds.drop('Life expectancy ', axis=1)
        y_train = train_ds['Life expectancy ']
        logger.debug('Model Training Started')
        pipe = Train_model(X_train, y_train, parameters)
        logger.debug('Model Training Completed successfully')

        save_pipeline(pipe)

        with Live(save_dvc_exp=True) as live:
            live.log_params(parameters)
        logger.debug('Parameters Log with For Experiment Tracking')

    except Exception as e:
        logger.error(f"❌ Model training pipeline failed: {e}")


if __name__ == "__main__":
    main()
