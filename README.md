# Intrusion Detection System (IDS) via Machine Learning (ML)

## Mohammadreza Zafarpour

This repository contains the code and resources for developing and evaluating machine learning-based Intrusion Detection Systems (IDS) using the ISCX NSL-KDD dataset 2009.

## Dataset
The ISCX NSL-KDD dataset 2009 is used for training and testing the machine learning models. It is an enhanced version of the KDD Cup 99 dataset, with a more balanced distribution of attack types and removal of redundant records. The dataset can be found in the data directory.

## Prerequisites
- Python 3.11
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- xgboost
- tensorflow
- keras

## Installation
- Clone the repository
- Install the required dependencies:
- pip install -r requirements.txt

## The following machine learning models are implemented and evaluated:
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Neural Network (LSTM)
- XGBoost (XGB)


## Results
The evaluation results, including accuracy, AUC, precision, recall, and F1-score, are stored in the results directory. The best-performing model is highlighted, and the results are visualized using charts and graphs.

## Acknowledgements
The ISCX NSL-KDD dataset 2009 is sourced from the Information Security Centre of Excellence (ISCX) at the University of New Brunswick, Canada.
The implementation of machine learning models is based on the scikit-learn, tensorflow, and keras libraries.
