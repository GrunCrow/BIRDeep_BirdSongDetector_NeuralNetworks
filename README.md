# BIRDeep Neural Networks

BIRDeep Audio Classifier repository, part of the BIRDeep project, aimed at classifying bird songs in audio recordings.

## Project Overview
This repository is a collection of neural networks and models designed for the BIRDeep project. Our primary objective is to develop an audio classifier capable of distinguishing between bird songs and background noise, and subsequently categorizing bird songs by their species.

## Repository Structure

- **`audio_classifier`**: This directory contains the core structure and files for the audio classifier.

- **`audio_detector`**: This directory contains the core structure and files for the audio detector.

- **`configs`**: In this section, you'll find configuration files that are crucial for the proper functioning of the classifier.

- **`Dataset`**: This directory holds the dataset, organized in the structure expected by our models.

- **`Data`**: Here, you'll find various files and information related to the dataset, including metadata and relevant details.

- **`Scripts`**: This section is dedicated to scripts that aid in data preparation, preprocessing, and other essential tasks for the project.

## Setting up the Conda Environment

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

