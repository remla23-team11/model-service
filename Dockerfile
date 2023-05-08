FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /root

# Clone the model-training repository
RUN apt-get update && \
    apt-get install -y git && \
    git clone https://github.com/remla23-team11/model-training.git /root/model-training

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install dependencies
COPY requirements.txt /root/
RUN pip install -r requirements.txt

# Copy the pre-processing code, trained model, and other necessary files
# from the cloned repository to the desired locations within the Docker image
RUN cp /root/model-training/preprocessing.py /root && \
    cp /root/model-training/c2_Classifier_Sentiment_Model /root && \
    cp /root/model-training/c1_BoW_Sentiment_Model.pkl /root

# Copy the Flask app
COPY app.py .

# Expose the port
EXPOSE 8080

# Start the Flask app
CMD [ "python", "app.py" ]
