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
    padded_sequences = sequence.pad_sequences(url_int_tokens, maxlen=150)
    return padded_sequences  # (n_samples, 150)



import numpy as np

def create_X_2(temp_X_2):
    # Drop the 'url' column and convert to float
    x = temp_X_2.drop(columns=['url']).values.astype(float)
    
    # Ensure there are exactly 15 features
    num_features = x.shape[1]  # Count existing features
    print('#################################',x.shape)
    if num_features < 15:
        # Pad with zeros to reach 15 features
        padding = np.zeros((x.shape[0], 15 - num_features))
        x = np.hstack((x, padding))  
    elif num_features > 15:
        # If more than 15, trim extra columns
        x = x[:, :15]  
    
    print('@@@@@@@@@@@@@@@@@@2',x.shape)
    # Ensure the final shape is (batch_size, 15, 1)
    return x.reshape(x.shape[0], 15, 1)



def construct_model():
    # Define named input layers
    input_A = Input(shape=(150,), name='main_input')        # Input for URL (X_1)
    input_B = Input(shape=(15, 1), name='main_input_2')      # Input for other features (X_2)

    # Load model_A and model_B
    model_A = load_model('models/model_A.h5', compile=False)
    model_B = load_model('models/model_B.h5', compile=False)

    # Chop off final layers to get feature outputs
    model_A = Model(inputs=model_A.input, outputs=model_A.layers[-2].output)
    model_B = Model(inputs=model_B.input, outputs=model_B.layers[-2].output)

    # Pass named inputs through the models
    output_A = model_A(input_A)
    output_B = model_B(input_B)

    # Merge and add final output layer
    merged = Add()([output_A, output_B])
    final_output = Dense(1, activation='sigmoid')(merged)

    # Create the full model
    model = Model(inputs=[input_A, input_B], outputs=final_output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predict_classes(model, x):
    proba = model.predict(x)
    return (proba > 0.5).astype('int32')

def load_and_preprocess_data():
    legitimate_train = pd.read_csv('features/legitimate_train.csv')
    legitimate_test = pd.read_csv('features/legitimate_test.csv')
    phish_train = pd.read_csv('features/phish_train.csv')
    phish_test = pd.read_csv('features/phish_test.csv')

    # Apply the scaling function to the features
    train = create_scaler(pd.concat([legitimate_train, phish_train], axis=0)).sample(frac=1).reset_index(drop=True)
    test = create_scaler(pd.concat([legitimate_test, phish_test], axis=0)).sample(frac=1).reset_index(drop=True)

    X_train = train.copy()
    X_test = test.copy()

    y_train = X_train.pop('result_flag')
    y_test = X_test.pop('result_flag')

    return X_train, y_train, X_test, y_test

# =========== MAIN TRAINING PROCESS ==========
if __name__ == "__main__":  # ✅ Correct
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    # Load sub-models
    model_A = load_model('models/model_A.h5', compile=False)
    model_A = Model(inputs=model_A.inputs, outputs=model_A.layers[-2].output)

    model_B = load_model('models/model_B.h5', compile=False)
    model_B = Model(inputs=model_B.inputs, outputs=model_B.layers[-2].output)

    # Early stopping & model checkpoint
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint('models/tmp_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # Create and train model
    model = construct_model()
    history = model.fit(
        [create_X_1(X_train), create_X_2(X_train)], y_train, 
        validation_split=0.1, epochs=10, batch_size=64, verbose=1, callbacks=[es, mc]
    )

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
