#!/bin/bash

# Downloads the SSW60 dataset:
# https://github.com/visipedia/ssw60
#
# 2022 Benjamin Kellenberger / Adapted by Elijah Cole

destFolder=datasets

mkdir -p $destFolder
echo "Downloading images..."
wget https://ml-inat-competition-datasets.s3.amazonaws.com/ssw60/ssw60.tar.gz -P $destFolder/.

echo "Removing the videos and images (and associated metadata), since we just want the audio..."
rm -r $destFolder/ssw60/video_ml
rm -r $destFolder/ssw60/video_ml.csv
rm -r $destFolder/ssw60/images_nabirds
rm -r $destFolder/ssw60/images_nabirds.csv
rm -r $destFolder/ssw60/images_inat
rm -r $destFolder/ssw60/images_inat.csv

echo "Unzipping..."
tar -xvf $destFolder/ssw60.tar.gz -C $destFolder/.

echo "Cleaning up..."
rm $destFolder/ssw60.tar.gz