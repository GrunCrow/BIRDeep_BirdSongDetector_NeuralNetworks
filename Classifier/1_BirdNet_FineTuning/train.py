# Load tensorflow model saved in ../BirdNET_GLOBAL_6K_V2.4_Model and predict on the audios of the csv: test.csv with path and label

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# Load the model
model = load_model("../BirdNET_GLOBAL_6K_V2.4_Model")

# Load the test data
test_data = pd.read_csv("test.csv")

# Load the audio files
audio_files = test_data['path'].values
audio_labels = test_data['label'].values

# Load the audio files and predict
predictions = []
for audio_file in audio_files:
    # Load the audio file
    audio = tf.io.read_file(audio_file)
    audio, _ = tf.audio.decode_wav(audio)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.expand_dims(audio, axis=0)
    
    # Predict
    prediction = model.predict(audio)
    predictions.append(np.argmax(prediction))

# Calculate accuracy
accuracy = np.mean(np.array(predictions) == audio_labels)
print("Accuracy: ", accuracy)