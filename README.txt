# Contents
This file describes the contents of this zip file.

## papers
Contains all papers cited in the main report.

## Python files
### face_blurring.py
Contains the blurring scripts and configs for the face pixelation and the black bar filter. Output file is configured in this file as well as the config for the blurring filters. This script reads in the reference images and outputs anonymised images to be used in extract_features.py.

### extract_features.py
Contains the feature extractor that uses DeepFace. File names to read and parse are configured at the top of the file. This script reads in image files and outputs feature vectors to be used in face_compare.py. The output is put into a folder called representations.

### face_compare.py
Takes in the output files from extract_features.py and compares the difference between the probe and anonymised/reference vectors. It outputs two file for each comparison. One containing mated scores and one containing non-mated scores.

## output (folder)
Contains anonymised versions of the reference images.

## probe (folder)
Contains the probe images from the color FERET database.

## reference (folder)
Contains the reference images from the color FERET database.

## representations (folder)
Contains the feature vector representations of the images from the color FERET database as well as the anonymised versions of the images.