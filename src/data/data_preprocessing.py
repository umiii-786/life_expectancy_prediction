import pandas as pd
import os
import json
from src.logging_config import logger

def load_data(folder_path: str)->tuple:
    try:
        train_ds = pd.read_csv(os.path.join(folder_path, 'train.csv'))
        test_ds = pd.read_csv(os.path.join(folder_path, 'test.csv'))
        logger.debug(f"loaded data Successfully")
        return train_ds, test_ds
    except FileNotFoundError as e:
        logger.error(f"CSV files not found in {folder_path}")
        raise
    #  FileNotFoundError(f"CSV files not found in {folder_path}") from e
    except Exception as e:
        logger.error("Error while loading datasets")
        raise
    #  RuntimeError("Error while loading datasets") from e


def load_country_region(path: str)-> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
        logger.debug(f"loaded country region json file Successfully")
        
    except FileNotFoundError as e:
        logger.error(f"JSON file not found: {path}")
        raise
    #  FileNotFoundError(f"JSON file not found: {path}") from e
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON format")
        raise 
    # ValueError("Invalid JSON format") from e


def fill_values_on_region_basis(group:pd.Series)->pd.Series:
    return group.fillna(group.median())


def fillNumericCols_on_Country_Basis(group:pd.Series)->pd.Series:
    return group.fillna(group.median())


def fill_missing_year(group:pd.Series)->pd.Series:
    try:
        years = group.to_numpy().copy()
        for i in range(len(years)):
            if pd.isna(years[i]):
                years[i] = 2015 if i == 0 else years[i - 1] - 1
        return pd.Series(years, index=group.index)
    except Exception as e:
        logger.error("Error while filling missing years")
        raise 


def handling_columns_with_low_frequency_missing(null_values_columns:list, low_missing:list, df:pd.DataFrame)->pd.DataFrame:
    try:
        for col in null_values_columns:
            df[col] = df.groupby('Country ')[col].transform(fillNumericCols_on_Country_Basis)

        for col in low_missing:
            df[col] = df.groupby('region')[col].transform(fill_values_on_region_basis)

        logger.debug("Handled missing values of column with less frequency")
        return df
    except KeyError as e:
        logger.error("Column not found during missing value handling")
        raise 


def handle_indicator_column_missing(indicator_cols:list, df:pd.DataFrame)->pd.DataFrame:
    try:
        for col in indicator_cols:
            df[col + '_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(df[col].median())
        logger.debug("Successfully handled indicator Columns")
        return df
    except KeyError as e:
        logger.error("Indicator column missing in dataframe")
        raise 


def apply_preprocessing(df:pd.DataFrame)->pd.DataFrame:
    try:
        df = df.dropna(subset=['Life expectancy '])
        df['Country '] = df['Country '].str.strip().str.lower()
        df['Year'] = df.groupby('Country ')['Year'].transform(fill_missing_year)

        null_values = df.isnull().sum()
        null_values_columns = null_values[null_values != 0].index

        country_region_map = load_country_region("country-X-region.json")
        df['region'] = df['Country '].map(country_region_map)

        indicator_cols = ['Hepatitis B', 'Income composition of resources',
                          'Schooling', 'GDP', 'Population']
        low_missing = ['Alcohol', ' BMI ', 'Total expenditure',
                       ' thinness  1-19 years', ' thinness 5-9 years']

        df = handling_columns_with_low_frequency_missing(null_values_columns, low_missing, df)
        df = handle_indicator_column_missing(indicator_cols, df)
        logger.debug('Successfully done Preprocessing')
        return df
    except Exception as e:
        logger.error("Preprocessing failed")
        raise 


def save_data(folder_path:str, train_ds:pd.DataFrame, test_ds:pd.DataFrame)->None:
    try:
        os.makedirs(folder_path, exist_ok=True)
        train_ds.to_csv(os.path.join(folder_path, 'preproced_train.csv'), index=False)
        test_ds.to_csv(os.path.join(folder_path, 'preproced_test.csv'), index=False)
        logger.debug('saved procecced data successfully')

    except Exception as e:
        logger.error("Failed to save processed data")
        raise 
    # RuntimeError() from e


def main()->None:
    try:
        load_path = os.path.join('data', 'raw')
        train_ds, test_ds = load_data(load_path)

        preproceed_train_ds = apply_preprocessing(train_ds)
        preproceed_test_ds = apply_preprocessing(test_ds)

        save_path = os.path.join('data', 'interim')
        save_data(save_path, preproceed_train_ds, preproceed_test_ds)
        logger.debug('Done All the Things Successfully')
    except Exception as e:
        logger.error("Pipeline execution failed")
        raise 


if __name__ == '__main__':
    main()
