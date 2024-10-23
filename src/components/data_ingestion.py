# Import necessary libraries
import os  # For file and directory operations
import sys  # For system-specific parameters and functions
import pandas as pd  # For data manipulation and analysis
from dataclasses import dataclass  # For creating configuration data classes

from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets

# Import components from the project
from src.components.data_transformation import DataTransformation, DataTransformationConfig  # Data transformation components
from src.components.model_tranier import ModelTrainer, ModelTrainerConfig  # Model training components
from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Logging utility

import sys
import os

# Add the src directory to Python's path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))



# Configuration for data ingestion
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

# Data Ingestion Class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Initialize with default configuration

    def initiate_data_ingestion(self):
        logging.info('Data ingestion process started.')
        try:
            # Load dataset
            df = pd.read_csv('notebook/data/data_preprocessing.csv')
            logging.info('Dataset loaded successfully into a pandas DataFrame.')

            # Create directories for data storage if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            # Split the dataset into training and testing sets
            logging.info('Initiating train-test split.')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training and testing sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion completed successfully.')

            # Return paths to the train and test data
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error('Exception occurred during data ingestion.')
            raise CustomException(e, sys)

# Main process execution
if __name__ == '__main__':
    # Perform data ingestion
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    # Perform data transformation
    DataTransformationConfig=DataTransformationConfig()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Model training (uncomment when ModelTrainer is ready to be used)el
    ModelTrainerConfig=ModelTrainerConfig()
    model_trainer = ModelTrainer()
    metrics = model_trainer.initiate_model_training(train_arr, test_arr)
