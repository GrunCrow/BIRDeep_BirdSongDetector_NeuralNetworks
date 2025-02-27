# BIRDeep Bird Song Detector by Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14940480.svg)](https://doi.org/10.5281/zenodo.14940480)

BIRDeep Bird Song Detector repository, part of the BIRDeep project, aimed at detecting bird songs in audio recordings.

This repository contains the codes, data, and projects associated with the research paper "Decoding the Sounds of Doñana: Advancements in Bird Detection and Identification Through Deep Learning." This project focuses on leveraging deep learning techniques to improve bird species identification from audio recordings collected in Doñana National Park.

The dataset used in this research is available at a [Hugging Face Repository](https://huggingface.co/datasets/GrunCrow/BIRDeep_AudioAnnotations).

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

Passive acoustic monitoring (PAM) is an essential tool for biodiversity conservation, but it generates vast amounts of audio data that are challenging to analyze. This project aims to automate bird species detection using a multi-stage deep learning approach. We combined a YOLOv8-based Bird Song Detector with a fine-tuned BirdNET model to improve species classification accuracy.

## Repository Structure

The repository is organized as follows:

- `Bird Classifiers/`: Contains the codes and outputs of the bird classifiers used in the project. It includes BirdNET classifier, embeddings for machine learning based classifiers and other deep learning architectures.
  - `BirdNET/`: Contains BirdNET generated models, training plots and predictions by some of the different models tested.
  - `models/`: Contains the final classifiers used in the project.
  - `Scripts/`: Scripts used for data generation and training of the classifiers. Evaluation scripts are all together in general `Scripts/` folder.
- `BIRDeep Song Detector/`: This directory contains the core structure and files for the Bird Song Detector. Contains the trainings and pre-trained and fine-tuned models data of the Bird Song Detector.
  - `runs/detect/`: Output files, including model predictions and performance metricsfrom the Bird Song Detector.
- `Data/`: Contains the audio data and annotations used for training and evaluation, you can check the [BIRDeep_AudioAnnotations Dataset](https://huggingface.co/datasets/GrunCrow/BIRDeep_AudioAnnotations). Also generated images for Bird Song Detector and Deep Learning Classifiers.
- `Research/`: Information collected during literature review, only a base research README missing a lot of information, for more, please, go to manuscripts.
- `Scripts/`: Jupyter notebooks for data preprocessing and exploratory data analysis.
- `README.md`: This file.

## Data

### Audio Recordings

The audio recordings were collected using AudioMoth devices in Doñana National Park. The recordings are organized by habitat type (marshland, scrubland, and ecotone) and location. Each recording is a 1-minute segment sampled every 10 minutes, with a sampling rate of 32 kHz.

### Annotations

Expert annotators labeled 461 minutes of audio data, identifying bird vocalizations and other relevant sounds. Annotations are provided in a standard format with start time, end time, and frequency range for each bird vocalization.

If more information about the Data is needed, please, refer to the [Data repository](https://huggingface.co/datasets/GrunCrow/BIRDeep_AudioAnnotations).

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

## Research

### Abstract

Passive Acoustic Monitoring (PAM) that uses devices like automatic audio recorders has become a fundamental tool in conserving and managing natural ecosystems. However, this practice generates a large volume of unsupervised audio data, and extracting valid information for environmental monitoring is a significant challenge. It is then critically necessary to use methods that leverage Deep Learning techniques for automating species detection. BirdNET is a model trained for bird identification that has succeeded in many study systems, especially in North America and central-Northern Europe, but it results inadequate for other regions or local studies due to insufficient training and its bias on focal sounds rather than entire soundscapes. Another added problem for bird species detection is that many audios recorded in PAM programs are empty of sounds of species of interest or these sounds overlap. To overcome these problems, we present here a multi-stage process for automatically identifying bird vocalizations applied to a local study site with concerning conservation threats for bird populations, Doñana National Park (SW Spain). Our working pipeline included first the development of a YOLOv8-based Bird Song Detector, and second, the fine-tuning of BirdNET for species classification at the local scale. We annotated 461 minutes of audio data from three main habitats across nine different locations within Doñana, resulting in 3749 annotations representing 38 different classes. Mel spectrograms were employed as graphical representations of bird audio data, facilitating the application of image processing methods. Several detectors were trained in different experiments, which included data augmentation and hyperparameter exploration to improve the model’s robustness. The model giving the best results included the creation of synthetic background audios with data augmentation and the use of an environmental sound library (ESC-50 dataset). This proposed pipeline using the Bird Song Detector as a first step for detecting segments with bird vocalizations and applying later a fine-tuning of BirdNET for the local bird species of Doñana significantly improved BirdNET detections; True Positives were increased by 281.97%, and False Negatives were reduced by 62.03%. Our approach demostrated then to be effective for bird species identification at a particular local study. These findings underscore the importance of adapting general-purpose tools to address specific challenges in biodiversity monitoring, as is the case of Doñana. Automatically detecting bird species serves for tracking the health status of this threatened ecosystem, given the sensitivity of birds to environmental changes, and helps in the design of conservation measures for reducing biodiversity loss.

### Introduction

#### Main Objetive

The main objective of the research was to develop a pipeline that optimized the detection and classification of bird species in audio recordings using Deep Learning techniques. This pipeline allows the annotation of these recordings to be automated, facilitating the performance of relevant ecological studies.

#### Context Background

Relevant background information includes the use of passive acoustic monitoring techniques for the conservation and management of natural ecosystems. In addition, Deep Learning models such as BirdNET have been used for bird identification, although these models have not presented satisfactory results for the specific ecological context of Doñana, such as the most abundant species or the soundscape.

#### Research Question

The main hypothesis of the research is that the development of a Deep Learning model specifically trained with data from Doñana will significantly improve the detection and classification of bird species compared to existing general models. Transfer Learning techniques will be applied to existing models for the classification of bird species in audio recordings, trying to minimize analysis times, improve current performance and adapt it to the specific characteristics of the Doñana case study.

### Material and Methods

#### Methods to Data Collection

Data was collected using automatic audio recording devices (AudioMoths) in three different habitats in Doñana National Park. Approximately 500 minutes of audio data were recorded. There are 9 recorders in 3 different habitats, which are constantly running, recording 1 minute and leaving 9 minutes between recordings. That is, 1 minute is recorded for every 10 minutes. The recordings were made prioritising those times when the birds are most active in order to try to have as many audio recordings of songs as possible, specifically a few hours before dawn until midday.

The name of the places correspond to the following recorders (included as metadata in CSVs of the dataset) and coordinates:

| Number | Habitat    | Place Name        | Recorder | Lat        | Lon          | Installation Date |
|--------|------------|-------------------|----------|------------|--------------|-------------------|
| Site 1 | low shrubland | Monteblanco       | AM1      | 37.074     | -6.624       | 03/02/2023        |
| Site 2 | high shrubland | Sabinar           | AM2      | 37.1869444 | -6.720555556 | 03/02/2023        |
| Site 3 | high shrubland | Ojillo            | AM3      | 37.2008333 | -6.613888889 | 03/02/2023        |
| Site 4 | low shrubland | Pozo Sta Olalla   | AM4      | 37.2202778 | -6.729444444 | 03/02/2023        |
| Site 5 | ecotone    | Torre Palacio     | AM8      | 37.1052778 | -6.5875      | 03/02/2023        |
| Site 6 | ecotone    | Pajarera          | AM10     | 37.1055556 | -6.586944444 | 03/02/2023        |
| Site 7 | ecotone    | Caño Martinazo    | AM11     | 37.2086111 | -6.512222222 | 03/02/2023        |
| Site 8 | marshland  | Cancela Millán    | AM15     | 37.0563889 | -6.6025      | 03/02/2023        |
| Site 9 | marshland  | Juncabalejo       | AM16     | 36.9361111 | -6.378333333 | 03/02/2023        |

#### Data Analysis Procedure

The audio data was transformed into Mel spectrograms, which were then used to train a Deep Learning model. First, a detector was developed to find time windows in which a bird song is detected. Then, BirdNET was trained to create a classifier adapted to the ecological context of Doñana. The final objective is to use a pipeline in which the detector obtains the time windows in which there is a bird song and BirdNET, with fine-tuning, performs the classification of the species present.

#### Theory

The theory behind this methodology is that Deep Learning models can learn to identify and classify bird species from Mel spectrograms, which are graphical representations of audio data. Just as a general model can achieve good results when Transfer Learning is performed to adapt it to a specific problem.

According to the original BirdNET paper: "In summary, BirdNET achieved a mean average precision of 0.791 for single-species recordings, a F0.5 score of 0.414 for annotated soundscapes, and an average correlation of 0.251 with hotspot observation across 121 species and 4 years of audio data." That is, on audios that belong to the domain to which BirdNET belongs, in a real context in which the audios contain soundscapes, that is, soundscapes, the performance is not the best. On the other hand "The most common sources of false-positive detections were other vocalizing animals (e.g., insects, anurans, mammals), geophysical noise (e.g., wind, rain, thunder), human vocal and non-vocal sounds (e.g., whistling, footsteps, speech), anthropogenic sounds typically encountered in urban areas (e.g., cars, airplanes, si rens), and electronic recorder noise. The Google AudioSet is one of the largest collections of human-labeled sounds that span a wide range of classes that are organized in an ontology (Gemmeke et al., 2017). BirdNET can produce many false positives, creating a bird song detector step beforehand can reduce the number of false positives. Following an idea from DeepFaune, in which a first step based on Megadetector is established for photo-trapping cameras to eliminate empty images from those containing animals and thus be able to subsequently apply a classifier only on those samples that are True Positive, reducing the number of False Positives in the classifier.

### Theory/Calculation

#### Methodological approach

The methodology is based on the premise that Deep Learning models, trained with Mel spectrograms, can effectively identify and classify bird species in audio recordings. Transfer Learning allows a general model to be adapted to a specific dataset, improving its performance in the new context.

#### Calculation from theory

Practical computations involve converting audio data into Mel spectrograms, training a preliminary bird song detector, and fine-tuning BirdNET for species classification. Model performance metrics are evaluated against annotated datasets to measure improvement.

#### Equations and Mathematical Models used

The core mathematical model involves Convolutional Neural Networks (CNN) to process images, in this case, graphical representations of audio recordings through Mel spectrograms. For the detector, the YOLOv8 architecture is used. For the classifier, BirdNET is used, using Transfer Learning techniques to fine-tune BirdNET to specific ecological data.

### Results

The most significant results are that it seems that there was not enough data available to generate a robust detection model. Future work is needed to improve the detector, since after carrying out various experiments, achieving improvements has been difficult. The greatest improvement achieved has been by moving from temporal and frequency detections to only temporal detections, including the entire frequency spectrum for training and waiting for the entire frequency spectrum for detections.

In addition to finding difficulties with empty instances, i.e. True Negatives and False Positives, Data Augmentation techniques have been included to reduce this. First, background audios were edited for training, modifying intensity and adding noise. This improved, but not significantly. Later, audios from the ESC-50 library were included, which contains focal sounds, eliminating the sounds of birds such as crows and chickens. After applying the training, first results were obtained in which the network did not learn and ended up classifying all instances as empty due to the disproportion of ESC-50 audios compared to the dataset of interest. The number of ESC-50 audios was reduced to find a balance and thus the results were improved, although not very significantly.

The best detector model achieves a mAP50 of 0.29756 in the train, in validation it was around X.XX (to be completed) and in test it was similar to the validation.

To measure the performance of BirdNET, specific functions were created to measure the metrics of the classifications made on the test, allowing the conf_score and the IoU to be adjusted.

### Discussion

The results underscore the importance of tailoring deep learning models to specific ecological contexts for accurate species identification. This study demonstrates that general models such as BirdNET can be significantly improved by specific tuning with contextual data.

Compared to previous studies using BirdNET, this research shows an improvement in species detection and classification in a specific ecological context by incorporating a preliminary detector and fine-tuning the model.

Limitations include the relatively small size of the dataset and potential performance improvement with larger and more diverse data. The ability of the model to generalize to other ecological contexts also requires further investigation.

Future research should focus on expanding the dataset, testing the pipeline in different ecological contexts, and exploring additional tuning techniques to further improve model performance.

### Conclusions

The main conclusion is that fine-tuning deep learning models with context-specific data significantly improves accuracy and efficiency in bird species detection and classification. This study highlights the need for tailored approaches in echoacoustic monitoring.

The findings have important implications for biodiversity monitoring, suggesting that tailored deep learning models can provide more accurate and efficient tools for ecological studies.

Recommendations include developing larger and more diverse datasets for training, applying the pipeline in various ecological contexts, and exploring advanced fine-tuning techniques to further improve performance.

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

