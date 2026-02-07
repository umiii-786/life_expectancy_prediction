import pandas as pd
import os
from src.logging_config import logger

def load_data(path: str) -> tuple:
    try:
        train_path = os.path.join(path, 'handled_train.csv')
        test_path = os.path.join(path, 'handled_test.csv')

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.error("Required input CSV files are missing")
            raise 

        train_ds = pd.read_csv(train_path)
        test_ds = pd.read_csv(test_path)
        logger.debug('Loaded data Successfully ')
        return train_ds, test_ds

    except pd.errors.EmptyDataError:
        logger.error("One or more CSV files are empty")
        raise 
    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise 


def buildFeatures(df: pd.DataFrame) -> pd.DataFrame:
    try:
        required_cols = ['Alcohol', ' BMI ', 'Total expenditure', 'Population']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise 

        # Avoid division by zero
        if (df[' BMI '] == 0).any():
            logger.error("BMI column contains zero values")
            raise 

        if (df['Population'] == 0).any():
            logger.error("Population column contains zero values")
            raise 

        df['Alcohol_Consumption_Index'] = df['Alcohol'] / df[' BMI ']
        df['Health_Spend_per_Capita'] = df['Total expenditure'] / df['Population']
        logger.debug('Created Columns Successfully')
        return df

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise 


def save_data(path: str, train_ds: pd.DataFrame, test_ds: pd.DataFrame) -> None:
    try:
        os.makedirs(path, exist_ok=True)

        train_ds.to_csv(
            os.path.join(path, 'new_feature_train.csv'), index=False
        )
        test_ds.to_csv(
            os.path.join(path, 'new_feature_test.csv'), index=False
        )
        logger.debug('Saved data with new Features Files Successfully')

    except PermissionError:
        logger.error("Permission denied while saving output files")
        raise 
    except Exception as e:
        logger.error(f"Error while saving data: {e}")
        raise 


def main() -> None:
    try:
        load_path = os.path.join('data', 'processed')
        train_ds, test_ds = load_data(load_path)

        train_ds = buildFeatures(train_ds)
        test_ds = buildFeatures(test_ds)

        save_path = os.path.join('data', 'feature_engineering', 'feature_creation')
        save_data(save_path, train_ds, test_ds)

        logger.debug("✅ Feature Creation completed successfully")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
