import numpy as np
import pandas as pd
import os
from string import printable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Add, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_scaler(df):
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

def construct_model():
    mergedOut = Add()([model_A.output, model_B.output])
    mergedOut = Dense(1, activation='sigmoid')(mergedOut)
    model = Model([model_A.input, model_B.input], mergedOut)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predict_classes(model, x):
    proba = model.predict(x)
    return (proba > 0.5).astype('int32')

# Load data
legitimate_train = pd.read_csv('features/legitimate_train.csv')
legitimate_test = pd.read_csv('features/legitimate_test.csv')
phish_train = pd.read_csv('features/phish_train.csv')
phish_test = pd.read_csv('features/phish_test.csv')

train = create_scaler(pd.concat([legitimate_train, phish_train], axis=0)).sample(frac=1).reset_index(drop=True)
test = create_scaler(pd.concat([legitimate_test, phish_test], axis=0)).sample(frac=1).reset_index(drop=True)

X_train, y_train = train.drop(columns=['result_flag']), train.result_flag
X_test, y_test = test.drop(columns=['result_flag']), test.result_flag

# Load sub-models
model_A = load_model('models/model_A.h5')
model_A = Model(inputs=model_A.inputs, outputs=model_A.layers[-2].output)

model_B = load_model('models/model_B.h5')
model_B = Model(inputs=model_B.inputs, outputs=model_B.layers[-2].output)

# Early stopping & model checkpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('models/tmp_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# Create and train model
model = construct_model()
history = model.fit([create_X_1(X_train), create_X_2(X_train)], y_train, validation_split=0.1, epochs=500, batch_size=64, verbose=1, callbacks=[es, mc])

# Save training history
with open("train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Load and save the best model
model = load_model('models/tmp_model.h5')
model.save('models/model_C.h5')
os.remove('models/tmp_model.h5')

# Evaluate model performance
y_pred = predict_classes(model, [create_X_1(X_test), create_X_2(X_test)])
print(confusion_matrix(y_test, y_pred))
print("All done.")
