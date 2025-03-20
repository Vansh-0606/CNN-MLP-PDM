# Detecting phishing attacks using a combined model of LSTM and CNN.
This repository implements a novel phishing detection approach that combines LSTM and CNN models.

# Folder Structure:
features/ → Pre-extracted features from a dataset.
models/ → Contains pre-trained models:
model_A.h5 and model_B.h5: Trained on a dataset of 40,000 samples.
model_C.h5: A combined, trained model.
const_data.py → Extracts features from HTML pages (instructions in the script).
train.py → Trains a new model after feature extraction.
evaluate.py → Tests and evaluates a trained model on test data.

#  Feature Extraction
Run const_data.py to extract features from web pages.

# Model Training
After feature extraction, train a model using:
python train.py

# Model Evaluation
Evaluate the trained model on test data using:
python evaluate.py
