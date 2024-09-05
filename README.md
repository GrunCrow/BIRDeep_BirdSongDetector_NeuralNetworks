# BIRDeep Bird Song Detector by Neural Networks

BIRDeep Audio Classifier repository, part of the BIRDeep project, aimed at classifying bird songs in audio recordings.

This repository contains the codes, data, and projects associated with the research paper "Decoding the Sounds of Doñana: Advancements in Bird Detection and Identification Through Deep Learning." This project focuses on leveraging deep learning techniques to improve bird species identification from audio recordings collected in Doñana National Park.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Passive acoustic monitoring (PAM) is an essential tool for biodiversity conservation, but it generates vast amounts of audio data that are challenging to analyze. This project aims to automate bird species detection using a multi-stage deep learning approach. We combined a YOLOv8-based Bird Song Detector with a fine-tuned BirdNET model to improve species classification accuracy.


## Repository Structure

The repository is organized as follows:
- `BIRDeep/`: Contains the trainings and pre-trained and fine-tuned models data of the Bird Song Detector.
- `BirdNET/`: Contains the segments and structure needed to classify audios by BirdNET, BirdNET pre-trained and fine-tuned models and metadata and BirdNET predictions.
- `Data/`: Contains the audio data and annotations used for training and evaluation, you can check the [BIRDeep_AudioAnnotations Dataset](https://huggingface.co/datasets/GrunCrow/BIRDeep_AudioAnnotations).
- `Detector`: This directory contains the core structure and files for the Bird Song Detector.
- `ESC-50`: ESC-50 complete and original dataset as cloned from the repository.
- `Extras`: Tasks related to the dataset as plots that are not directly related to the Deep Learning models development.
- `Research/`: Information collected during literature review.
- `Scripts/`: Jupyter notebooks for data preprocessing and exploratory data analysis.
- `Web/`: Friendly web-app to use the model.
- `runs/detect/`: Output files, including model predictions and performance metricsfrom the Bird Song Detector.
- `README.md`: This file.

## Data

### Audio Recordings

The audio recordings were collected using AudioMoth devices in Doñana National Park. The recordings are organized by habitat type (marshland, scrubland, and ecotone) and location. Each recording is a 1-minute segment sampled every 10 minutes, with a sampling rate of 32 kHz.

### Annotations

Expert annotators labeled 461 minutes of audio data, identifying bird vocalizations and other relevant sounds. Annotations are provided in a standard format with start time, end time, and frequency range for each bird vocalization.

## Models

### Bird Song Detector

The Bird Song Detector is based on YOLOv8 and is trained to identify temporal windows containing bird vocalizations. The model can detect bird songs even from species not encountered during training.

### BirdNET Classifier

BirdNET is a deep learning model specifically designed for bird species classification. We fine-tuned BirdNET V2.4 to improve its performance on the audio data from Doñana National Park. The model processes audio segments identified by the Bird Song Detector to classify bird species.

## Usage

### Prerequisites

- Python 3.8 or higher
- Required Python packages (listed in `environment.yml`)

### Setting up the Conda Environment

If you want to reproduce this project, you can start by setting up the Conda environment. Follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/GrunCrow/BIRDeep_NeuralNetworks
    ```

2. Navigate to the project's directory:
    ```
    cd BIRDeep_NeuralNetworks
    ```

3. Create a Conda environment using the provided environment.yml file:

    ```
    conda env create -f environment.yml
    ```

    This will create a Conda environment named "BIRDeep" with the required dependencies.

4. Activate the Conda environment:

    ```
    conda activate BIRDeep
    ```

You are now ready to work with the project using the provided environment.

## Results

The proposed pipeline significantly improves bird species identification by increasing True Positives and reducing False Negatives.

## Research

### Abstract

Passive acoustic monitoring through the use of devices such as automatic audio recorders has emerged as a fundamental tool in the conservation and management of natural ecosystems. However, this practice presents a significant challenge given it generates a large volume of data that does not have human supervision. In order to obtain valid information for ecoacoustics studies, the main bottleneck now is to manage large datasets of acoustic recordings for identifying species of interest.  Automated species detection methods using deep learning techniques are paramount for this. It is presented a multi-stage process for automatic analysis of bird recordings from Doñana National Park (SW Spain) obtained through AudioMoths thanks to the BIRDeep project. Although existing Deep Learning models such as BirdNET have shown success in bird identification in other study systems, they did not present satisfactory results for the most abundant species of Doñana, likely due to inadequate training on Doñana’s specific data and its bias on focal sounds, rather than entire soundscapes. Consequently, we annotated about 500 minutes of audio data at three different habitats  and trained our own model.  By using the Mel spectrogram as a graphical representation of bird audio data, we show how this technique can be leveraged to apply image processing methods and computer vision in the analysis of acoustic data analysis. For this, it is critical the availability of labeled, high-quality datasets. In conclusion, our advances show that general-purpose tools may not always be the best solution in deep learning and ecoacoustics, emphasizing the importance of adapting these tools to the specific problem being addressed. By fine- tuning deep learning models and techniques to the unique characteristics of ecoacoustic data from a specific context, researchers can improve the accuracy and efficiency of biodiversity monitoring efforts.

### Introduction

**¿Cuál es el objetivo principal de tu investigación?**

El objetivo principal de la investigación es desarrollar un pipeline que optimice la detección y clasificación de especies de aves en grabaciones de audio utilizando técnicas de Deep Learning. Este pipeline permitirá automatizar la anotación de estas grabaciones, facilitando la realización de estudios ecológicos relevantes.

**¿Qué antecedentes relevantes debes incluir para situar el contexto de tu estudio?**

Los antecedentes relevantes incluyen el uso de técnicas de monitoreo acústico pasivo para la conservación y gestión de ecosistemas naturales. Además, se han utilizado modelos de Deep Learning como BirdNET para la identificación de aves, aunque estos modelos no han presentado resultados satisfactorios para el contexto ecológico específico de Doñana, como las especies más abundantes o el paisaje sonoro.

**¿Qué hipótesis o preguntas de investigación estás abordando?**

La hipótesis principal de la investigación es que el desarrollo de un modelo de Deep Learning específicamente entrenado con datos de Doñana mejorará significativamente la detección y clasificación de especies de aves en comparación con los modelos generales existentes. Se realizarán técnicas de Transfer Learning sobre modelos ya existentes para la clasificación de especies de aves en grabaciones de audio, intentando minimizar los tiempos de análisis, mejorar el rendimiento actual y adaptarlo a las características específicas del caso de estudio de Doñana.

### Material and Methods

**¿Qué métodos utilizaste para recolectar datos (p.ej., equipos, ubicaciones, tiempos)?**

Los datos se recogieron utilizando dispositivos de grabación automática de audio (AudioMoths) en tres hábitats diferentes del Parque Nacional de Doñana. Se anotaron aproximadamente 500 minutos de datos de audio. Hay 9 grabadoras en 3 hábitats diferentes, que están funcionando constantemente grabando 1 minuto y dejando 9 minutos entre grabación y grabación. Es decir, de cada 10 minutos se graba 1 minuto. Las anotaciones se han realizado priorizando aquellos tiempos en los que las aves tienen mayor actividad para intentar disponer de tantos audios con cantos como sea posible, específicamente unas horas antes del amanecer hasta mediodía.

**¿Qué procedimientos seguiste para analizar los datos?**

Los datos de audio se transformaron en espectrogramas de Mel, que luego se utilizaron para entrenar un modelo de Deep Learning. Primero se desarrolló un detector para encontrar ventanas temporales en las que se detecta el canto de un ave. Luego se entrenó BirdNET para crear un clasificador adaptado al contexto ecológico de Doñana. El objetivo final es crear un pipeline en el que el detector obtenga las ventanas temporales en las que hay un canto de ave y BirdNET, con fine-tuning, realice la clasificación de las especies presentes.

**¿Hay algún aspecto específico de la teoría que respalde tu metodología?**

La teoría que respalda esta metodología es que los modelos de Deep Learning pueden aprender a identificar y clasificar especies de aves a partir de espectrogramas de Mel, que son representaciones gráficas de los datos de audio. Así como un modelo general puede conseguir buenos resultados cuando se realiza Transfer Learning para adaptarlo a un problema específico.

Según el paper original de BirdNET: "In summary, BirdNET achieved a mean average precision of 0.791 for single-species recordings, a F0.5 score of 0.414 for annotated soundscapes, and an average correlation of 0.251 with hotspot observation across 121 species and 4 years of audio data." Es decir sobre audios que pertenecen al dominio sobre el que pertenece BirdNET, en un contexto real en el que los audios contienen paisaje sonoro, es decir, soundscapes, el rendimiento no es el mejor. Por otro lado "The most common sources of false-positive detections were other vocalizing animals (e.g., insects, anurans, mam mals), geophysical noise (e.g., wind, rain, thunder), human vocal and non-vocal sounds (e.g., whistling, footsteps, speech), anthropogenic sounds typically encountered in urban areas (e.g., cars, airplanes, si rens), and electronic recorder noise. The Google AudioSet is one of the largest collections of human-labeled sounds that span a wide range of classes that are organized in an ontology (Gemmeke et al., 2017). We used 16 classes from the AudioSet". BirdNET puede producir muchos falsos positivos, crear un paso previo de detector de cantos de aves puede reducir el número de falsos positivos. Siguiendo una idea de DeepFaune, en la que se establece para cámaras de fototrampeo un primer paso basado en Megadetector para eliminar las imágenes vacías de las que contienen animales y así poder aplicar posteriormente un clasificador solo sobre aquellas muestras que son True Positive, reduciendo el número de False Positives en el clasificador.

### Theory/Calculation

**¿Qué teoría subyace en tu enfoque metodológico?**
La metodología se basa en la premisa de que los modelos de Deep Learning, entrenados con espectrogramas de Mel, pueden identificar y clasificar eficazmente especies de aves en grabaciones de audios. El Transfer Learning permite adaptar un modelo general a un conjunto de datos específico, mejorando su rendimiento en el nuevo contexto.

**¿Cómo se desarrollan los cálculos prácticos a partir de la teoría?**
Los cálculos prácticos implican convertir datos de audio en espectrogramas de Mel, entrenar un detector preliminar de cantos de aves, y ajustar BirdNET para la clasificación de especies. Se evalúan las métricas de rendimiento del modelo en relación con conjuntos de datos anotados para medir la mejora.

**¿Hay alguna fórmula o modelo matemático específico que estés utilizando?**
El modelo matemático central implica Redes Neuronales Convolucionales (CNN) para procesar imágenes, en este caso, representaciones gráficas de grabaciones de audios a través de espectrogramas de Mel. Para el detector se usa la arquitectura YOLOv8. Para el clasificador se utiliza BirdNET, utilizando técnicas de Transfer Learning para ajustar BirdNET a datos ecológicos específicos.

### Results

**¿Cuáles son los resultados más significativos de tu estudio?**

Los resultados más significativos son que parece ser que no se disponían de suficientes datos como para generar un modelo de detección robusto. Es necesario desarrollar un trabajo futuro para mejorar el detector, ya que después de realizar diversos experimentos, conseguir mejoras ha sido complicado. La mayor mejora conseguida ha sido al pasar de detecciones temporales y frecuenciales a solo temporales, incluyendo todo el espectro de frecuencia para el entrenamiento y esperando todo el espectro de frecuencias para las detecciones.

Además de encontrar dificultades con las instancias vacías, es decir, True Negatives y False Positives, se han incluido técnicas de Data Augmentation para reducir esto. Primero se editaron audios de background para entrenamiento, modificando intensidad y añadiendo ruido. Se mejoró, pero no de manera muy significativa. Posteriormente se incluyeron audios de la librería ESC-50 que contiene sonidos focales, eliminando los sonidos de aves como cuervos y gallinas. Tras aplicar el entrenamiento, primero se obtuvieron resultados en los que la red no aprendía y terminaba clasificando como vacías todas las instancias debido a la desproporción de audios de ESC-50 respecto al dataset de interés. Se redujo el número de audios de ESC-50 para encontrar un balance y así se consiguió mejorar los resultados, aunque tampoco de manera muy significativa.

El mejor modelo de detector consigue en el train un mAP50 de 0.29756, en validación era en torno a X.XX (por completar) y en test similar a la validación.

Para medir el rendimiento de BirdNET se crearon funciones específicas para medir las métricas de las clasificaciones realizadas sobre test, permitiendo ajustar el conf_score y el IoU.

**¿Puedes proporcionar datos cuantitativos específicos (p.ej., gráficos, tablas)?**

Sí, a continuación se presentan los resultados cuantitativos obtenidos con diferentes configuraciones del modelo:

Detector (train mAP50: 0.29756, validación: X.XX, test: similar a validación):

BirdNET sin preentrenar (lista de especies de Doñana):

Métricas del Detector:
Accuracy: 0.1194
Precision: 1.0
Recall: 0.0781
F1-Score: 0.1449
Métricas del Detector + Clasificador:
Accuracy: 0.0677
Precision: 1.0
Recall: 0.0213
F1-Score: 0.0418
Otras:
False Positives: 0

BirdNET con lista de especies de las ground truth:

Métricas del Detector:
Accuracy: 0.1450
Precision: 0.9796
Recall: 0.1071
F1-Score: 0.1932
Métricas del Detector + Clasificador:
Accuracy: 0.0886
Precision: 0.95
Recall: 0.0453
F1-Score: 0.0866
Otras:
False Positives: 1

BirdNET tras fine-tuning (sin detector):

Métricas del Detector:
Accuracy: 0.0682
Precision: 1.0
Recall: 0.0246
F1-Score: 0.0479
Métricas del Detector + Clasificador:
Accuracy: 0.0562
Precision: 1.0
Recall: 0.0113
F1-Score: 0.0224
Otras:
False Positives: 0

BirdNET tras fine-tuning (conf_score threshold 0.1):

Métricas del Detector:
Accuracy: 0.4571
Precision: 0.8465
Recall: 0.4799
F1-Score: 0.6125
Métricas del Detector + Clasificador:
Accuracy: 0.2668
Precision: 0.6855
Recall: 0.2673
F1-Score: 0.3846
Otras:
False Positives: 39

BirdNET con detector (conf_score threshold 0.2):

Métricas del Detector:
Accuracy: 0.0768
Precision: 1.0
Recall: 0.0335
F1-Score: 0.0648
Métricas del Detector + Clasificador:
Accuracy: 0.0607
Precision: 1.0
Recall: 0.0159
F1-Score: 0.0313
Otras:
False Positives: 0

**¿Hay resultados que hayan sido inesperados o particularmente interesantes?**

Sí, uno de los resultados inesperados fue la dificultad significativa para mejorar el rendimiento del detector incluso después de varios experimentos y ajustes, especialmente en la reducción de instancias vacías. La implementación de técnicas de Data Augmentation y la inclusión de audios de la librería ESC-50 no mejoraron los resultados de manera significativa. Otro resultado interesante fue la necesidad de ajustar el conf_score threshold a valores más bajos (como 0.1) para obtener predicciones útiles con BirdNET tras el fine-tuning, lo cual subraya la sensibilidad del modelo a los umbrales de confianza.

### Discussion

**¿Qué significan tus resultados en el contexto de la investigación existente?**

Los resultados subrayan la importancia de adaptar los modelos de Deep Learning a contextos ecológicos específicos para una identificación precisa de especies. Este estudio demuestra que los modelos generales como BirdNET pueden mejorarse significativamente mediante ajustes específicos con datos contextuales.

**¿Cómo se comparan tus hallazgos con estudios previos?**

Comparado con estudios previos que usan BirdNET, esta investigación muestra una mejora en la detección y clasificación de especies en un contexto ecológico específico mediante la incorporación de un detector preliminar y el ajuste del modelo.

**¿Qué limitaciones identificas en tu estudio?**

Las limitaciones incluyen el tamaño relativamente pequeño del conjunto de datos y la potencial mejora del rendimiento con datos más extensos y diversos. La capacidad del modelo para generalizar a otros contextos ecológicos también requiere mayor investigación.

**¿Qué sugerencias tienes para investigaciones futuras?**

La investigación futura debe centrarse en expandir el conjunto de datos, probar el pipeline en diferentes contextos ecológicos y explorar técnicas adicionales de ajuste para mejorar aún más el rendimiento del modelo.

### Conclusions

**¿Cuáles son las principales conclusiones que se derivan de tu estudio?**

La conclusión principal es que ajustar los modelos de Deep Learning con datos específicos del contexto mejora significativamente la precisión y eficiencia en la detección y clasificación de especies de aves. Este estudio destaca la necesidad de enfoques personalizados en el monitoreo ecoacústico.

**¿Qué implicaciones tienen tus hallazgos para la práctica o la teoría?**

Los hallazgos tienen importantes implicaciones para el monitoreo de la biodiversidad, sugiriendo que los modelos de Deep Learning personalizados pueden proporcionar herramientas más precisas y eficientes para estudios ecológicos.

**¿Hay recomendaciones específicas que se deriven de tu investigación?**

Las recomendaciones incluyen desarrollar conjuntos de datos más grandes y diversos para el entrenamiento, aplicar el pipeline en varios contextos ecológicos y explorar técnicas avanzadas de ajuste para mejorar aún más el rendimiento.

<!--## Getting Started

Before using the audio classifier, make sure to follow these steps:

1. Install the necessary dependencies by running the provided setup script.

2. Download the dataset and place it in the `Dataset` directory, adhering to the expected structure.

3. Configure the classifier by editing the files in the `configs` directory as needed.

4. Run the preprocessing scripts available in the `Scripts` section to prepare the data for training.

## Usage

Once you've completed the setup, you can use the audio classifier to detect bird songs and classify them by species. Detailed instructions on how to use the classifier can be found in the `audio_classifier` directory.-->

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Stay tuned for updates and advancements in our pursuit to understand and classify bird songs more accurately with the help of deep learning and neural networks.

