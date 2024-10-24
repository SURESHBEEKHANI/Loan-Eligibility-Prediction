import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

import pandas as pd

class PredictPipeline:
    def __init__(self):
        """Initialize the PredictPipeline class."""
        pass

    def predict(self, features):
        """
        Predict the target variable using the pre-trained model.

        Args:
            features (pd.DataFrame): Input features for prediction.

        Returns:
            np.ndarray: Model prediction results.
        """
        try:
            # Load the preprocessor and model from the artifact paths
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            # Apply preprocessing to the input features
            data_scaled = preprocessor.transform(features)

            # Make predictions with the model
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            logging.error('Error occurred during prediction', exc_info=True)
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, self_employed, education, no_of_dependents, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value):
        self.self_employed = self_employed
        self.education = education
        self.no_of_dependents = no_of_dependents
        self.income_annum = income_annum
        self.loan_amount = loan_amount
        self.loan_term = loan_term
        self.cibil_score = cibil_score
        self.residential_assets_value = residential_assets_value
        self.commercial_assets_value = commercial_assets_value
        self.luxury_assets_value = luxury_assets_value
        self.bank_asset_value = bank_asset_value

    def get_data_as_dataframe(self):
        """
        Converts the input data into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame representing the input features.
        """
        try:
            data = {
                'self_employed': [self.self_employed],
                'education': [self.education],
                'no_of_dependents': [self.no_of_dependents],
                'income_annum': [self.income_annum],
                'loan_amount': [self.loan_amount],
                'loan_term': [self.loan_term],
                'cibil_score': [self.cibil_score],
                'residential_assets_value': [self.residential_assets_value],
                'commercial_assets_value': [self.commercial_assets_value],
                'luxury_assets_value': [self.luxury_assets_value],
                'bank_asset_value': [self.bank_asset_value]
            }

            df = pd.DataFrame(data)

            # Log DataFrame creation
            logging.info('DataFrame created successfully')
            logging.debug(f'DataFrame contents: {df}')  # Use debug for detailed output
            return df
        except Exception as e:
            logging.error('Error occurred while creating DataFrame', exc_info=True)
            raise CustomException(e, sys)
