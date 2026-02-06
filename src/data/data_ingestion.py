import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
from src.logging_config import logger

# ------------------ Functions ------------------
def loadYml(file_path: str) -> float:
    try:
        if not os.path.exists(file_path):
            logger.error(f"YAML file not found: {file_path}")
            raise 

        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        logger.debug(f"loaded yml file successfully: {file_path}")


        if "data_ingestion" not in config:
            logger.error("'data_ingestion' key missing in params.yaml")
            raise 

        if "test_size" not in config['data_ingestion']:
            logger.error("'testsize' key missing in data-ingestion.")
            raise

        logger.debug("'returned test size successfully")
        return config["data_ingestion"]['test_size']

    except Exception as e:
        logger.error(f"Error while loading YAML file: {e}")
        raise


def load_data(data_path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(data_path):
            logger.error(f"CSV file not found: {data_path}")
            raise
        #  FileNotFoundError(f"CSV file not found: {data_path}")

        df = pd.read_csv(data_path)

        if df.empty:
            logger.error("Loaded dataset is empty")
            raise 

        logger.debug(f"Successfully loaded the dataset")
        return df

    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise


def save_data(path: str, train_ds: pd.DataFrame, test_ds: pd.DataFrame) -> None:
    try:
        os.makedirs(path, exist_ok=True)

        train_path = os.path.join(path, "train.csv")
        test_path = os.path.join(path, "test.csv")

        train_ds.to_csv(train_path, index=False)
        test_ds.to_csv(test_path, index=False)

        logger.debug(f"Train data saved at: {train_path}")
        logger.debug(f"Test data saved at: {test_path}")

    except Exception as e:
        logger.error(f"Error while saving data: {e}")
        raise


# ------------------ Main Pipeline ------------------
def main() -> None:
    try:
        logger.info("Starting data ingestion pipeline")

        df = load_data(r"C:\Users\M.UMAIR\Desktop\Taknofest Task\data.csv")

        test_size = loadYml("params.yaml")
        # test_size = config.get("test_size")

        if not (0 < test_size < 1):
            logger.error("test_size must be between 0 and 1")
            raise

        train_ds, test_ds = train_test_split(
            df,
            test_size=test_size,
            random_state=2,
            shuffle=True
        )

        path = os.path.join("data", "raw")
        save_data(path, train_ds, test_ds)

        logger.debug("Data ingestion completed successfully")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
