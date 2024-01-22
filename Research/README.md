# Bird Acoustics Analysis Project

## Project Objective
This project aims to analyze bird acoustics using different audio libraries and pre-trained models. The goal is to gain insights that can be applied in the field of ecology.

The audio libraries mentioned provide a vast amount of bird sound recordings, which can be used for training and testing models. The pre-trained models can be used as a starting point, potentially using transfer learning techniques to adapt these models to specific tasks or datasets.

## Audio Libraries

### [Xeno Canto](https://xeno-canto.org/)
Xeno Canto is a website dedicated to sharing bird sounds from all over the world. It is a collaborative project inviting users to share their own recordings, help identify mystery recordings, or share their expertise in the forums. The dataset covers the sounds of the bird sound collection of Xeno-canto (XC). [Xeno Canto Dataset](https://www.gbif.org/dataset/b1047888-ae52-4179-9dd5-5448ea342a24) provides access to sound recordings of wildlife from around the world.
- Around 700.000 audios (not all are publicly available)
- Annotations are done by the community and can be changed trough time
- You can filter by fields, for example, area.
    - In Doñana there are around 2.500 annotated audios.
- Downloadable Data is a CSV and the media is a link if it is available.
- Media can be audio or images.

### [Macaulay Library](https://macaulaylibrary.org/)
The Macaulay Library is a scientific archive for research, education, and conservation. The [Macaulay Dataset](https://www.gbif.org/dataset/7f6dd0f7-9ed4-49c0-bb71-b2a9c7fed9f1) includes:
- +175,000 audio recordings covering 75 percent of the world's bird species, with an ever-increasing number of insect, fish, frog, and mammal recordings as well

## Pre-trained Models

### Perch
Bird classification
 - [Xeno Canto Dataset](https://www.gbif.org/dataset/b1047888-ae52-4179-9dd5-5448ea342a24). Audios from California, Hawaii, NY, Peru...
 - EfficientNetB1
 - Torch

### [BirdNET](https://birdnet.cornell.edu/)
BirdNET is a research platform that aims at recognizing birds by sound at scale. It is the platform BIRDeep is currently using.
- 6.522 worlwide classes including different bird species and non birds
- You can filter by area so predictable classes are reduced
- MobileNet V1

### [MixIT](https://github.com/google-research/sound-separation/blob/master/models/bird_mixit/README.md)
Separate audio into channels potentially representing separate sources. There are different models, one of them has been trained on birds data

### [YAMNet](https://www.kaggle.com/models/google/yamnet/frameworks/tensorFlow2/variations/yamnet/versions/1?tfhub-redirect=true)
Embedding model trained on AudioSet YouTube, not specific for birds.
- AudioSet-Youtube
- 521 classes, such as laughter, barking, or a siren
- Tensorflow
- Runned on WAV files
- Gives 4/8 outputs
- Gives start / end times

## Other Tools

### Ribbit
Detect sounds with periodic pulsing patterns.


