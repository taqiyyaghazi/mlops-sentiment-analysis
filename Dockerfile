FROM tensorflow/serving:latest

COPY ./output/serving_model /models/sentiment-model

ENV MODEL_NAME=sentiment-model
ENV PORT=8501 