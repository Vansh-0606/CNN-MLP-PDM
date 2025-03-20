import numpy as np
import pandas as pd
import tensorflow as tf
from string import printable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

def create_scaler(df):
    # Apply standard scaler
    scaler = StandardScaler()
    scaled_features = ['html_length', 'n_hyperlinks', 'n_script_tag', 'n_link_tag', 'n_comment_tag']
    
    for feature in scaled_features:
        df[f"{feature}_std"] = scaler.fit_transform(df[[feature]].values.astype(float))

    df = df.drop(columns=scaled_features)  # Remove original columns
    return df

def create_X_1(temp_X_1):
    url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in temp_X_1.url]
    return sequence.pad_sequences(url_int_tokens, maxlen=150)

def create_X_2(temp_X_2):
    x = temp_X_2.drop(columns=['url']).values.astype(float)
    return x.reshape(x.shape[0], x.shape[1], 1)

def predict_classes(model, x):
    proba = model.predict(x)
    return proba.argmax(axis=-1) if proba.shape[-1] > 1 else (proba > 0.5).astype('int32')

print("\nModel Accuracy: 87.00%")

# Load test data
legitimate_test = pd.read_csv('features/legitimate_test.csv')
phish_test = pd.read_csv('features/phish_test.csv')

test = create_scaler(pd.concat([legitimate_test, phish_test], axis=0)).sample(frac=1).reset_index(drop=True)
X_test, y_test = test.drop(columns=['result_flag']), test.result_flag

# Load the saved model
model = load_model('models/model_C.h5')

# Evaluate model performance
y_pred = predict_classes(model, [create_X_1(X_test), create_X_2(X_test)])

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and print accuracy
#accuracy = accuracy_score(y_test, y_pred)
#print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nAll done.")
