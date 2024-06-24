import gradio as gr
import librosa
import numpy as np
from PIL import Image
from ultralytics import YOLO
import shutil
import os
from io import BytesIO
import matplotlib.pyplot as plt

ROOT_PATH = "BIRDeep_NeuralNetworks/"

# Load the pre-trained YOLOv8 model for bird song detection from a local path
MODEL_WEIGHTS = ROOT_PATH + "BIRDeep/0_test6/weights/best.pt"

# Load YOLOv8 model with the specified weights
model = YOLO(MODEL_WEIGHTS)

# Function to convert audio clip to mel spectrogram image
def audio_to_mel_spectrogram(audio_clip):
    # Load the audio clip
    # audio_clip_path = process_file(audio_clip)

    y, sr = librosa.load(audio_clip, sr=None)

    # Define the frequency range
    fmin = 1  # Minimum frequency (0 Hz)
    fmax = 16000  # Maximum frequency (32000 Hz)

    fig, ax = plt.subplots(figsize=(12, 6))  # Set the background color to black
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", fmin=fmin, fmax=fmax, ax=ax)  # Specify frequency range
    ax.axis('off')  # Remove axes

    # Convert the plot to an image object
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    mel_spectrogram_image = Image.open(buf)

    return mel_spectrogram_image

def update_mel_spectrogram(audio_clip):
    mel_spectrogram_image = audio_to_mel_spectrogram(audio_clip)
    # Convert the image to PNG format before returning
    mel_spectrogram_image = mel_spectrogram_image.convert("RGB")
    return mel_spectrogram_image

def detect_bird_song(mel_spectrogram, confidence_threshold):
    #If runs folder exists, delete it
    if os.path.exists("runs"):
        shutil.rmtree("runs")

    # Perform object detection on the mel spectrogram image using the model
    results = model.predict(source=mel_spectrogram,  # (str, optional) source directory for images or videos
                             save=True,
                             conf=confidence_threshold,
                             save_txt=True,  # (bool) save results as .txt file
                             save_conf=True,  # (bool) save results with confidence scores
                             )

    # If there is a txt in any subdirectory of "runs", then read txt file else return "No Bird Songs Detected"
    for root, dirs, files in os.walk("runs"):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r") as file:
                    predictions = file.read()
                return predictions
    return "No Bird Songs Detected"

def process_predictions(predictions, mel_spectrogram_image):
    # Get dimensions of the mel spectrogram image
    width, height = mel_spectrogram_image.size
    
    # Initialize an empty list to store processed predictions
    processed_predictions = []
    
    # Parse predictions and convert bounding box coordinates
    for line in predictions.split("\n"):
        if line.strip() == "":
            continue
        
        # Parse prediction line
        class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, line.strip().split())
        
        # Convert bounding box coordinates to image coordinates
        x_center = x_center_norm * width
        y_center = y_center_norm * height
        box_width = width_norm * width
        box_height = height_norm * height
        
        # Convert image coordinates to time and frequency
        time_start = (x_center - box_width / 2) / width * 60  # Assuming 60 seconds
        time_end = (x_center + box_width / 2) / width * 60
        freq_start = ((height - y_center) - box_height / 2) / height * 8000  # Assuming 0 to 8000 Hz
        freq_end = ((height - y_center) + box_height / 2) / height * 8000
        
        # Format the processed prediction
        processed_prediction = f"Bird Detected: {time_start:.2f}s to {time_end:.2f}s From={freq_start:.2f}Hz to {freq_end:.2f}Hz"
        
        # Append processed prediction to the list
        processed_predictions.append(processed_prediction)
    
    return processed_predictions

def predict(audio_clip, confidence_threshold):
    mel_spectrogram_image = update_mel_spectrogram(audio_clip)
    predictions_result = detect_bird_song(mel_spectrogram_image, confidence_threshold)
    # processed_predictions = process_predictions(predictions_result, mel_spectrogram_image)
    return mel_spectrogram_image, predictions_result
    #return mel_spectrogram_image, "\n".join(processed_predictions)

# Create Gradio interface with custom layout
audio_clip = gr.Audio(label="Upload Audio Clip", type="filepath")
mel_spectrogram = gr.Image(label="Mel Spectrogram")
confidence_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.05)
predictions = gr.Textbox(label="Predictions", value="Predictions will be placed here")

iface = gr.Interface(
    fn=predict,
    inputs=[audio_clip, confidence_threshold],
    outputs=[mel_spectrogram, predictions],
    title="Bird Song Detector",
)

iface.launch()