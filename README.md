# BIRDeep Bird Song Detector by Neural Networks

BIRDeep Audio Classifier repository, part of the BIRDeep project, aimed at classifying bird songs in audio recordings.

This repository contains the codes, data, and projects associated with the research paper "Decoding the Sounds of Doñana: Advancements in Bird Detection and Identification Through Deep Learning." This project focuses on leveraging deep learning techniques to improve bird species identification from audio recordings collected in Doñana National Park.

The dataset used in this research is available at a [Hugging Face Repository](https://huggingface.co/datasets/GrunCrow/BIRDeep_AudioAnnotations).

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

## Results

The proposed pipeline significantly improves bird species identification by increasing True Positives and reducing False Negatives.

## Research

### Abstract

Passive acoustic monitoring (PAM) that uses devices like automatic audio recorders has become a fundamental tool in conserving and managing natural ecosystems. However, this practice generates a large volume of unsupervised audio data, and extracting valid information for environmental monitoring is a significant challenge. It is then critically necessary to use methods that leverage Deep Learning techniques for automating species detection. BirdNET is a model trained for bird identification that has succeeded in many study systems, especially in North America or Europe, but it results inadequate for other regions due to insufficient training and its bias on focal sounds rather than entire soundscapes. Another added problem for species detection is that many audios recorded in PAM programs are empty of sounds of species of interest or these sounds overlap. This study presents a multi-stage process for automatically identifying bird vocalizations that includes first a YOLOv8-based Bird Song Detector, and second, a fine-tuned BirdNET for species classification at a local scale with enhanced detection accuracy. As a study case, we applied this Bird Song Detector to audio recordings collected in Doñana National Park (SW Spain) as a part of the BIRDeep project. We annotated 461 minutes of audio data from three main habitats across nine different locations within Doñana, resulting in 3749 annotations representing 38 different classes. Mel spectrograms were employed as graphical representations of bird audio data, facilitating the application of image processing methods. Several detectors were trained in different experiments, which included data augmentation and hyperparameter exploration to improve the model’s robustness. The model giving the best results included the creation of synthetic background audios with data augmentation and the use of an environmental sound library. This proposed pipeline using the Bird Song Detector as a preliminary step, significantly improves BirdNET detections by increasing True Positives by approximately 281.97%, and reducing False Negatives by about 62.03%, thus demonstrating a novel and effective approach for bird species identification. Our findings underscore the importance of adapting general-purpose tools to address specific challenges in biodiversity monitoring. The experimental results show that fine-tuning Deep Learning models that account for the unique characteristics of specific ecological contexts can substantially enhance the accuracy and efficiency of PAM's efforts.

### Introduction

**Main Objetive**

The main objective of the research was to develop a pipeline that optimized the detection and classification of bird species in audio recordings using Deep Learning techniques. This pipeline allows the annotation of these recordings to be automated, facilitating the performance of relevant ecological studies.

**Context Background**

Relevant background information includes the use of passive acoustic monitoring techniques for the conservation and management of natural ecosystems. In addition, Deep Learning models such as BirdNET have been used for bird identification, although these models have not presented satisfactory results for the specific ecological context of Doñana, such as the most abundant species or the soundscape.

**Research Question**

The main hypothesis of the research is that the development of a Deep Learning model specifically trained with data from Doñana will significantly improve the detection and classification of bird species compared to existing general models. Transfer Learning techniques will be applied to existing models for the classification of bird species in audio recordings, trying to minimize analysis times, improve current performance and adapt it to the specific characteristics of the Doñana case study.

### Material and Methods

**Methods to Data Collection**

Data was collected using automatic audio recording devices (AudioMoths) in three different habitats in Doñana National Park. Approximately 500 minutes of audio data were recorded. There are 9 recorders in 3 different habitats, which are constantly running, recording 1 minute and leaving 9 minutes between recordings. That is, 1 minute is recorded for every 10 minutes. The recordings were made prioritising those times when the birds are most active in order to try to have as many audio recordings of songs as possible, specifically a few hours before dawn until midday.

The name of the places correspond to the following recorders (included as metadata in CSVs of the dataset) and coordinates:
- Monteblanco
- Ojillo
- Sabinar
- Caño Martinazo
- Cancela Millán
- Pajarera
- Torre Palacio
- Pozo Santa Olalla
- Juncabalejo

**Data Analysis Procedure**

The audio data was transformed into Mel spectrograms, which were then used to train a Deep Learning model. First, a detector was developed to find time windows in which a bird song is detected. Then, BirdNET was trained to create a classifier adapted to the ecological context of Doñana. The final objective is to use a pipeline in which the detector obtains the time windows in which there is a bird song and BirdNET, with fine-tuning, performs the classification of the species present.

**Theory**

The theory behind this methodology is that Deep Learning models can learn to identify and classify bird species from Mel spectrograms, which are graphical representations of audio data. Just as a general model can achieve good results when Transfer Learning is performed to adapt it to a specific problem.

According to the original BirdNET paper: "In summary, BirdNET achieved a mean average precision of 0.791 for single-species recordings, a F0.5 score of 0.414 for annotated soundscapes, and an average correlation of 0.251 with hotspot observation across 121 species and 4 years of audio data." That is, on audios that belong to the domain to which BirdNET belongs, in a real context in which the audios contain soundscapes, that is, soundscapes, the performance is not the best. On the other hand "The most common sources of false-positive detections were other vocalizing animals (e.g., insects, anurans, mam mals), geophysical noise (e.g., wind, rain, thunder), human vocal and non-vocal sounds (e.g., whistling, footsteps, speech), anthropogenic sounds typically encountered in urban areas (e.g., cars, airplanes, si rens), and electronic recorder noise. The Google AudioSet is one of the largest collections of human-labeled sounds that span a wide range of classes that are organized in an ontology (Gemmeke et al., 2017). BirdNET can produce many false positives, creating a bird song detector step beforehand can reduce the number of false positives. Following an idea from DeepFaune, in which a first step based on Megadetector is established for photo-trapping cameras to eliminate empty images from those containing animals and thus be able to subsequently apply a classifier only on those samples that are True Positive, reducing the number of False Positives in the classifier.

### Theory/Calculation

**Methodological approach**
The methodology is based on the premise that Deep Learning models, trained with Mel spectrograms, can effectively identify and classify bird species in audio recordings. Transfer Learning allows a general model to be adapted to a specific dataset, improving its performance in the new context.

**Calculation from theory**
Practical computations involve converting audio data into Mel spectrograms, training a preliminary bird song detector, and fine-tuning BirdNET for species classification. Model performance metrics are evaluated against annotated datasets to measure improvement.

**Equations and Mathematical Models used**
The core mathematical model involves Convolutional Neural Networks (CNN) to process images, in this case, graphical representations of audio recordings through Mel spectrograms. For the detector, the YOLOv8 architecture is used. For the classifier, BirdNET is used, using Transfer Learning techniques to fine-tune BirdNET to specific ecological data.

### Results

The most significant results are that it seems that there was not enough data available to generate a robust detection model. Future work is needed to improve the detector, since after carrying out various experiments, achieving improvements has been difficult. The greatest improvement achieved has been by moving from temporal and frequency detections to only temporal detections, including the entire frequency spectrum for training and waiting for the entire frequency spectrum for detections.

In addition to finding difficulties with empty instances, i.e. True Negatives and False Positives, Data Augmentation techniques have been included to reduce this. First, background audios were edited for training, modifying intensity and adding noise. This improved, but not significantly. Later, audios from the ESC-50 library were included, which contains focal sounds, eliminating the sounds of birds such as crows and chickens. After applying the training, first results were obtained in which the network did not learn and ended up classifying all instances as empty due to the disproportion of ESC-50 audios compared to the dataset of interest. The number of ESC-50 audios was reduced to find a balance and thus the results were improved, although not very significantly.

The best detector model achieves a mAP50 of 0.29756 in the train, in validation it was around X.XX (to be completed) and in test it was similar to the validation.

To measure the performance of BirdNET, specific functions were created to measure the metrics of the classifications made on the test, allowing the conf_score and the IoU to be adjusted.

Detector (train mAP50: 0.29756, validación: X.XX, test: similar to validation):

#### BirdNET without fine-tuning (Doñana specie list):

Detector:
Accuracy: 0.1194
Precision: 1.0
Recall: 0.0781
F1-Score: 0.1449

Detector + Classifier:
Accuracy: 0.0677
Precision: 1.0
Recall: 0.0213
F1-Score: 0.0418

Others:
False Positives: 0

#### BirdNET con lista de especies de las ground truth:

Detector:
Accuracy: 0.1450
Precision: 0.9796
Recall: 0.1071
F1-Score: 0.1932

Detector + Classifier:
Accuracy: 0.0886
Precision: 0.95
Recall: 0.0453
F1-Score: 0.0866

Others:
False Positives: 1

#### BirdNET tras fine-tuning (sin detector):

Detector:
Accuracy: 0.0682
Precision: 1.0
Recall: 0.0246
F1-Score: 0.0479

Detector + Classifier:
Accuracy: 0.0562
Precision: 1.0
Recall: 0.0113
F1-Score: 0.0224

Others:
False Positives: 0

#### BirdNET tras fine-tuning (conf_score threshold 0.1):

Detector:
Accuracy: 0.4571
Precision: 0.8465
Recall: 0.4799
F1-Score: 0.6125

Detector + Classifier:
Accuracy: 0.2668
Precision: 0.6855
Recall: 0.2673
F1-Score: 0.3846

Others:
False Positives: 39

#### BirdNET with detector (conf_score threshold 0.2):

Detector:
Accuracy: 0.0768
Precision: 1.0
Recall: 0.0335
F1-Score: 0.0648

Detector + Classifier:
Accuracy: 0.0607
Precision: 1.0
Recall: 0.0159
F1-Score: 0.0313

Others:
False Positives: 0

One of the unexpected results was the significant difficulty in improving the detector performance even after several experiments and fine-tuning, especially in reducing empty instances. Implementing Data Augmentation techniques and including audios from the ESC-50 library did not significantly improve the results. Another interesting result was the need to adjust the conf_score threshold to lower values ​​(such as 0.1) to obtain useful predictions with BirdNET after fine-tuning, which underlines the sensitivity of the model to confidence thresholds.

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

