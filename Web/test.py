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

# Create Gradio interface with custom layout
audio_clip = gr.Audio(label="Upload Audio Clip", type="filepath")
mel_spectrogram = gr.Image(label="Mel Spectrogram")

def update_mel_spectrogram(audio_clip):
    mel_spectrogram_image = audio_to_mel_spectrogram(audio_clip)
    # Convert the image to PNG format before returning
    mel_spectrogram_image = mel_spectrogram_image.convert("RGB")
    return mel_spectrogram_image

iface = gr.Interface(
    fn=update_mel_spectrogram,
    inputs=audio_clip,
    outputs=mel_spectrogram,
    title="Bird Song Detector",
)

iface.launch()