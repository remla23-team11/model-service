#!/usr/bin/env python
# coding: utf-8

import pickle
import joblib
from flask import Flask, request, jsonify
from preprocessing import clean_review
from flasgger import Swagger
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

# Load ML model and pre-processing pipeline
cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', 'rb'))
model = joblib.load('c2_Classifier_Sentiment_Model')

# Define endpoint for making predictions
@app.route('/', methods=['POST'])
def predict():
    """
    Make a prediction
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                msg:
                    type: string
                    example: We are so glad we found this place.
    responses:
      200:
        description: Some result
    """
    data = request.json['msg']
    preprocessed_data = clean_review(data)
    vectorized_data = cv.transform([preprocessed_data]).toarray()
    predictions = model.predict(vectorized_data).tolist()
    return {'predictions': predictions}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
    
