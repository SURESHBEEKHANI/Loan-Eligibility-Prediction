from flask import Flask, render_template, request
import logging
import pandas as pd
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData
import numpy as np
from src.logger import logging

# Initialize the Flask application
app = Flask(__name__)


@app.route('/')
def home():
    # Render the main form page (index.html)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and validate data from form
        self_employed = request.form.get('self_employed')
        education = request.form.get('education')
        no_of_dependents = request.form.get('no_of_dependents')
        income_annum = request.form.get('income_annum')
        loan_amount = request.form.get('loan_amount')
        loan_term = request.form.get('loan_term')
        cibil_score = request.form.get('cibil_score')
        residential_assets_value = request.form.get('residential_assets_value')
        commercial_assets_value = request.form.get('commercial_assets_value')
        luxury_assets_value = request.form.get('luxury_assets_value')
        bank_asset_value = request.form.get('bank_asset_value')

        # Ensure no values are missing
        if not all([self_employed, education, no_of_dependents, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]):
            return render_template('index.html', prediction="Please fill out all required fields.")

        # Convert values to appropriate types
        no_of_dependents = int(no_of_dependents)
        income_annum = float(income_annum)
        loan_amount = float(loan_amount)
        loan_term = int(loan_term)
        cibil_score = int(cibil_score)
        residential_assets_value = float(residential_assets_value)
        commercial_assets_value = float(commercial_assets_value)
        luxury_assets_value = float(luxury_assets_value)
        bank_asset_value = float(bank_asset_value)

        # Create CustomData object
        data = CustomData(
            self_employed=self_employed,
            education=education,
            no_of_dependents=no_of_dependents,
            income_annum=income_annum,
            loan_amount=loan_amount,
            loan_term=loan_term,
            cibil_score=cibil_score,
            residential_assets_value=residential_assets_value,
            commercial_assets_value=commercial_assets_value,
            luxury_assets_value=luxury_assets_value,
            bank_asset_value=bank_asset_value
        )

        # Convert data to DataFrame
        features = data.get_data_as_dataframe()

        # Log the features
        logging.debug(f"Data passed to model: {features}")

        # Predict
        pipeline = PredictPipeline()
        prediction = pipeline.predict(features)

        # Determine result message
        prediction_result = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
        if prediction_result == 0:
            return render_template('index.html', prediction="Your loan application is approved!")
        else:
            return render_template('index.html', prediction="Unfortunately, your loan application was not approved.")

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return render_template('index.html', prediction=f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)