import csv
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

# Función para analizar un audio dado su path y otros datos
def analyze_audio(path, date_str):
    date = datetime.strptime(date_str, "%Y/%m/%d")
    recording = Recording(
        analyzer,
        path,
        date=date,
        min_conf=0.25,
        lat=37,
        lon=-6
    )
    recording.analyze()
    return recording.detections

# Función para analizar una lista de audios dada un archivo CSV
def analyze_audio_list(csv_file, output_csv):
    with open(csv_file, newline='') as csvfile, open(output_csv, 'w', newline='') as outputfile:
        reader = csv.DictReader(csvfile)
        fieldnames = ['path', 'date', 'start_time', 'end_time', 'detections']
        writer = csv.DictWriter(outputfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            path = row['path']
            date = row['date']
            detections = analyze_audio(path, date)
            writer.writerow({'path': path, 'date': date,'detections': detections})

# Llama a la función para analizar la lista de audios y guardar los resultados en otro archivo CSV
analyze_audio_list('test.csv', 'resultados_audios.csv')
