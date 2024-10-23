import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from ..exception import CustomException
from src.logger import logging
from src.utils import save_object

# Configuration for saving the preprocessor object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

# Data Transformation Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        Creates and returns the preprocessing object with pipelines for numerical and categorical columns.
        '''
        try:
            # Specify the columns for transformation
            categorical_cols = ['self_employed', 'education']
            numerical_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                              'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 
                              'bank_asset_value']
            
            # Define categories for ordinal encoding
            self_employed_categories = ['No', 'Yes']
            education_categories = ['Graduate', 'NotGraduate']

            # Numerical pipeline: Handle missing values and scale
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
                ('scaler', StandardScaler())  # Scale numerical features
            ])

            # Categorical pipeline: Handle missing values, ordinal encode, and scale
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with the most frequent value
                ('ordinal_encoder', OrdinalEncoder(categories=[self_employed_categories, education_categories])),  # Ordinal encode
                ('scaler', StandardScaler())  # Scale categorical features
            ])

            logging.info(f'Categorical Columns: {categorical_cols}')
            logging.info(f'Numerical Columns: {numerical_cols}')

            # Combine numerical and categorical pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.error('Exception occurred in get_data_transformation_object method')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        Initiates the data transformation process on training and testing datasets.
        '''
        try:
            # Load training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Loaded training and testing datasets successfully.')
            logging.debug(f'Train DataFrame Head: \n{train_df.head()}')
            logging.debug(f'Test DataFrame Head: \n{test_df.head()}')

            # Obtain the preprocessing object
            preprocessing_obj = self.get_data_transformation_object()

            # Specify target column
            target_column_name = 'loan_status'
            drop_columns = [target_column_name]

            # Separate input and target features for both training and testing datasets
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on the datasets.")

            # Apply the preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with the target column for both train and test sets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object for future use
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)
            logging.info('Preprocessor object saved successfully.')

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error('Exception occurred in initiate_data_transformation method')
            raise CustomException(e, sys)
