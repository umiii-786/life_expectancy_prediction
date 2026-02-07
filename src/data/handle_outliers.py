# using Z-score method for normal distribution columns
from scipy.stats import zscore
import os
import pandas as pd
from src.logging_config import logger

non_skewed_Cols = ['Schooling', ' BMI ', 'Total expenditure']
skewedColumns = [
    ' thinness  1-19 years', 'Population', 'GDP', 'Diphtheria ',
    'Polio', 'under-five deaths ', 'Measles ', 'Hepatitis B',
    'Alcohol', 'infant deaths', 'Adult Mortality',
    'percentage expenditure', ' HIV/AIDS'
]


def load_data(path: str) -> tuple:
    try:
        train_path = os.path.join(path, 'preproced_train.csv')
        test_path = os.path.join(path, 'preproced_test.csv')

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.error("Train or Test CSV file not found")
            raise 

        train_ds = pd.read_csv(train_path)
        test_ds = pd.read_csv(test_path)

        logger.debug("loaded_data Successfully")
        return train_ds, test_ds

    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
        raise 
    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise 


# IQR method for skewed numerical distributions (Country-wise Winsorization)
def imputebyIQR(group: pd.Series) -> pd.Series:
    try:
        Q1 = group.quantile(0.25)
        Q3 = group.quantile(0.75)
        IQR = Q3 - Q1

        if pd.isna(IQR) or IQR == 0:
            return group  # avoid invalid clipping

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return group.clip(lower, upper)

    except Exception as e:
        logger.error(f"IQR computation failed for column {group.name}: {e}")
        raise
    #  RuntimeError(f"IQR computation failed for column {group.name}: {e}") from e


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if 'Country ' not in df.columns:
            logger.error("Required column 'Country ' not found in DataFrame in Outlier Handling")
            raise 

        # Handle skewed columns using IQR
        for col in skewedColumns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
            df[col] = df.groupby('Country ')[col].transform(imputebyIQR)

        logger.debug('Imputed Outlier with IQR successfully')
        # Handle non-skewed columns using Z-score
        for col in non_skewed_Cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
            z = df.groupby('Country ')[col].transform(
                lambda x: zscore(x, nan_policy='omit')
            )
            df = df[z.abs() < 3]

        logger.debug('Imputed Outlier with Z-score successfully')

        return df

    except Exception as e:
        logger.error(f"Outlier handling failed: {e}")
        raise 
    # RuntimeError(f"Outlier handling failed: {e}") from e


def save_data(path: str, train_ds: pd.DataFrame, test_ds: pd.DataFrame) -> None:
    try:
        os.makedirs(path, exist_ok=True)

        train_ds.to_csv(os.path.join(path, 'handled_train.csv'), index=False)
        test_ds.to_csv(os.path.join(path, 'handled_test.csv'), index=False)
        logger.debug('Save outlier handled data Successfully')
    except PermissionError:
        logger.error("Permission denied while saving files")
        raise 
    except Exception as e:
        logger.error(f"Error while saving data: {e}")
        raise 


def main() -> None:
    try:
        load_path = os.path.join('data', 'interim')
        train_ds, test_ds = load_data(load_path)

        train_ds = handle_outliers(train_ds)
        test_ds = handle_outliers(test_ds)

        save_path = os.path.join('data', 'processed')
        save_data(save_path, train_ds, test_ds)

        print("✅ Outlier handling pipeline completed successfully")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")


if __name__ == "__main__":
    main()
