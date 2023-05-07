FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /root

# Upgrade pip
RUN python -m pip install --upgrade pip

# Clone the model-training repository
RUN apt-get update && \
    apt-get install -y git
RUN git clone https://github.com/remla23-team11/model-training.git

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt


# Copy the pre-processing code
COPY ./model-training/preprocessing.py .

# Copy the trained model
COPY ./model-training/c2_Classifier_Sentiment_Model .
COPY ./model-training/c1_BoW_Sentiment_Model.pkl .

# Copy the Flask app
COPY app.py .

# Expose the port
EXPOSE 8080

# Start the Flask app
CMD [ "python", "app.py" ]
