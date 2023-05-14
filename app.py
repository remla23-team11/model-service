#!/usr/bin/env python
# coding: utf-8

import pickle
import joblib
import prometheus_client
import time
from flask import Flask, request, jsonify, Response
from preprocessing import clean_review
from flasgger import Swagger
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Summary, generate_latest

# Initialize Flask app
app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Load ML model and pre-processing pipeline
cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', 'rb'))
model = joblib.load('c2_Classifier_Sentiment_Model')


# Initialize the metrics
# Counter metric to track the number of predictions made
prediction_counter = Counter('predictions_total', 'Total number of predictions made')
user_feedback = Counter('user_feedback_total', 'Total number of user feedback received', ['result'])

# Gauge
prediction_accuracy = prometheus_client.Gauge('prediction_accuracy', 'Accuracy of sentiment predictions')

correct_predictions = 0

# Define custom buckets
BUCKETS = [10, 50, 100, 150, 200]

# Initialize histogram with custom buckets
input_data_size_distribution = Histogram('input_data_size_distribution', 'Distribution of input data sizes', buckets=BUCKETS)

# Summary
sentiment_summary = Summary('sentiment_summary', 'Summary of sentiments predicted by the model', ['sentiment'])

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
  
    # Stage 1: Data preprocessing
    data = request.json['msg']
    preprocessed_data = clean_review(data)
    # Stage 2: Model prediction
    start_time = time.time()
    vectorized_data = cv.transform([preprocessed_data]).toarray()
    predictions = model.predict(vectorized_data).tolist()

    # Increment the prediction counter
    prediction_counter.inc()

    # Update input data size distribution histogram
    input_data_size_distribution.observe(len(request.json['msg']))

    # Track sentiment summary
    for prediction in predictions:
        sentiment_summary.labels(sentiment='positive' if prediction == 1 else 'negative').observe(1)

    return {'predictions': predictions}

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to provide feedback for.
          required: True
          schema:
            type: object
            required: 
              - msg
              - feedback
            properties:
                msg:
                    type: string
                    example: We are so glad we found this place.
                feedback:
                    type: string
                    example: correct
    responses:
      200:
        description: Feedback submitted successfully
    """

    global correct_predictions

    # Process the feedback 
    feedback = request.json['feedback']
    user_feedback.labels(result=feedback).inc()
    if feedback == 'correct':
        print(correct_predictions)
        correct_predictions += 1

    # Calculate accuracy
    total_predictions = prediction_counter._value.get()
    if total_predictions > 0:
      accuracy = correct_predictions / total_predictions
    else:
      accuracy = 0
    # # Update prediction accuracy gauge
    prediction_accuracy.set(accuracy)

    return {'message': 'Feedback submitted successfully'}


@app.route('/metrics', methods=['GET'])
def metrics():
    m = ""
  
    # Collect Counter metrics
    m += "# HELP predictions_total Total number of predictions made.\n"
    m += "# TYPE predictions_total counter\n"
    for metric in prediction_counter.collect():
        for sample in metric.samples:
            m += "{0} {1} {2}\n".format(sample.name, sample.value, sample.timestamp or '')

    # Counter: Collect user feedback Counter metrics
    m += "# HELP user_feedback_total Total number of user feedback received\n"
    m += "# TYPE user_feedback_total counter\n"
    m += generate_latest(user_feedback).decode('utf-8')

    # Gauge: Collect prediction accuracy Gauge metrics
    m += "# HELP prediction_accuracy Accuracy of sentiment predictions\n"
    m += "# TYPE prediction_accuracy gauge\n"
    for metric in prediction_accuracy.collect():
        for sample in metric.samples:
            m += "{0} {1} {2}\n".format(sample.name, sample.value, sample.timestamp or '')

    # Histogram: Input_data_size_distribution metric 
    m += generate_latest(input_data_size_distribution).decode('utf-8')

    # Summary: Sentiment_summary
    m += generate_latest(sentiment_summary).decode('utf-8')

    return Response(m, mimetype="text/plain")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
    
