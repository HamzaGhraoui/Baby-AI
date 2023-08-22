# Save this code in a file named app.py

import os
from flask import Flask, request, jsonify
from flask_cors import CORS ,cross_origin

import numpy as np
import librosa
import math
import tensorflow as tf

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5500"}})
# Load the trained model
from keras.models import load_model
model = load_model('Babies Cry Detection model.h5')

# Define mood_dict based on the classes of your model
mood_dict = {0: 'belly_pain', 1: 'burping', 2: 'discomfort', 3: 'hungry', 4: 'tired'}

# Function to process input audio and get prediction
def process_input(audio_file, track_duration):

    SAMPLE_RATE = 22050
    NUM_MFCC = 13
    N_FTT = 2048
    HOP_LENGTH = 512
    TRACK_DURATION = track_duration  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    NUM_SEGMENTS = 10

    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)

    mfcc_features = []

    for d in range(NUM_SEGMENTS):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(
            signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH
        )
        mfcc = mfcc.T

        mfcc_features.append(mfcc)

    # Padding to ensure all segments have the same shape
    max_length = max(len(segment) for segment in mfcc_features)
    mfcc_features_padded = np.zeros((NUM_SEGMENTS, max_length, NUM_MFCC))
    for i, segment in enumerate(mfcc_features):
        mfcc_features_padded[i, :segment.shape[0], :] = segment

    return mfcc_features_padded

@app.route('/')
def index():
    return "Baby Emotion Detector Backend"

@app.route('/predict', methods=['POST'])
@cross_origin()

def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            file_path = 'uploads/' + file.filename
            file.save(file_path)
            print("Received audio file:", file_path)

            mfcc_features = process_input(file_path, track_duration=5)

            # Convert to a 4D numpy array
            X_to_predict = np.expand_dims(mfcc_features, axis=3)

            # Perform prediction
            prediction = model.predict(X_to_predict)
            predicted_index = np.argmax(prediction, axis=1)
            predicted_mood = mood_dict[int(predicted_index[0])]

            print("Predicted mood:", predicted_mood)

            return jsonify({'predicted_mood': predicted_mood})
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(error_msg)
            return jsonify({'error': error_msg})

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=False)
