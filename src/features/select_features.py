import numpy as np
import pandas as pd
import os
import yaml
from src.logging_config import logger

def load_YML(path: str) -> float:
    try:
        if not os.path.exists(path):
            logger.error(f"YAML file not found: {path}")
            raise 
        # FileNotFoundError(f"YAML file not found: {path}")

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        cor_threshold = config['feature_selection']['threshold']
        logger.debug('fetched threshold values')
        return cor_threshold

    except KeyError:
        logger.error("Missing 'feature_selection.threshold' in params.yml")
        raise 
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML format: {e}")
        raise 
    except Exception as e:
        logger.error(f"Error loading YAML file: {e}")
        raise 


def load_data(path: str) -> tuple:
    try:
        train_path = os.path.join(path, 'new_feature_train.csv')
        test_path = os.path.join(path, 'new_feature_test.csv')
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.error("Feature-engineered CSV files are missing")
            raise 

        train_ds = pd.read_csv(train_path)
        test_ds = pd.read_csv(test_path)

        return train_ds, test_ds

    except pd.errors.EmptyDataError:
        logger.error("One or more CSV files are empty")
        raise 
    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise 
    # RuntimeError(f"Error while loading data: {e}") from e


def find_dropped_features(
    X: pd.DataFrame,
    y: pd.Series,
    cor_threshold: float,
    numerical_cols: list
) -> list:
    try:
        if X.empty or y.empty:
            logger.error("Input features or target is empty")
            raise 

        numeric_df = X[numerical_cols]

        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        high_corr_pairs = [
            (col1, col2)
            for col1 in upper_tri.columns
            for col2 in upper_tri.columns
            if upper_tri.loc[col1, col2] > cor_threshold
        ]

        target_corr = numeric_df.join(y).corr()[y.name].abs()
        to_drop = set()

        for f1, f2 in high_corr_pairs:
            if target_corr[f1] < target_corr[f2]:
                to_drop.add(f1)
            else:
                to_drop.add(f2)
        logger.debug('founded Column which is to dropped')
        return list(to_drop)

    except KeyError as e:
        logger.error(f"Column error during correlation computation: {e}")
        raise 
    except Exception as e:
        logger.error(f"Feature correlation analysis failed: {e}")
        raise 


def drop_features(df: pd.DataFrame, dropped_cols: list) -> pd.DataFrame:
    try:
        missing_cols = [col for col in dropped_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Columns to drop not found: {missing_cols}")
            raise 

        df = df.drop(columns=dropped_cols)

        if 'Country ' not in df.columns:
            logger.error("Required column 'Country ' not found")
            raise 

        df = df.drop('Country ', axis=1)
        logger.debug('dropped all the correalted features from the data set ')
        return df

    except Exception as e:
        logger.error(f"Feature dropping failed: {e}")
        raise 


def save_data(path: str, train_ds: pd.DataFrame, test_ds: pd.DataFrame) -> None:
    try:
        os.makedirs(path, exist_ok=True)

        train_ds.to_csv(
            os.path.join(path, 'selected_feature_train.csv'), index=False
        )
        test_ds.to_csv(
            os.path.join(path, 'selected_feature_test.csv'), index=False
        )

        logger.debug('saved data with dropped features successfully')
    except PermissionError:
        logger.error("Permission denied while saving selected features")
        raise 
    except Exception as e:
        logger.error(f"Error while saving data: {e}")
        raise 


def main() -> None:
    try:
        load_path = os.path.join(
            'data', 'feature_engineering', 'feature_creation'
        )
        save_path = os.path.join(
            'data', 'feature_engineering', 'feature_selection'
        )

        train_ds, test_ds = load_data(load_path)
        cor_threshold = load_YML('params.yaml')

        indicator_cols = [
            'Hepatitis B_missing',
            'Income composition of resources_missing',
            'Schooling_missing',
            'GDP_missing',
            'Population_missing',
            'Life expectancy '
        ]

        if 'Life expectancy ' not in train_ds.columns:
            logger.error("Target column 'Life expectancy ' not found")
            raise 

        X = train_ds.drop(columns=indicator_cols)
        y = train_ds['Life expectancy ']

        numerical_cols = X.select_dtypes(exclude='object').columns

        dropped_cols = find_dropped_features(
            X, y, cor_threshold, numerical_cols
        )

        train_ds = drop_features(train_ds, dropped_cols)
        test_ds = drop_features(test_ds, dropped_cols)

        save_data(save_path, train_ds, test_ds)

        logger.debug("✅ Feature selection completed successfully")

    except Exception as e:
        logger.error(f"❌ Feature selection pipeline failed: {e}")


if __name__ == "__main__":
    main()
