# Text Classification & Text Mining on Twenty Newsgroups Dataset

## Overview

This project applies text classification and text mining techniques on the Twenty Newsgroups dataset. The dataset consists of newsgroup posts categorized into 20 different topics, making it a popular benchmark for text analysis and machine learning tasks.

## Features
(1) Preprocessing: Tokenization, stopword removal, stemming/lemmatization, and TF-IDF vectorization.

(2) Exploratory Data Analysis (EDA): Word frequency analysis, word cloud visualization.

(3) Text Classification: Implementation of various machine learning models such as Naïve Bayes, Logistic Regression, and Support Vector Machines (SVM).

(4) Pattern-based Augmentation: Common text patterns are incorporated as variables to generate augmented data, leading to improved classification performance.

(5) Evaluation Metrics: Accuracy, precision, recall, F1-score, and confusion matrix visualization.

## Requirements

### Computing Resources
- Operating system: MacOS
- RAM: 8 GB
- Disk space: 8 GB

### Language:
- [Python 3+](https://www.python.org/download/releases/3.0/) (Note: coding will be done strictly on Python 3)
    - We are using Python 3.9.6.
    
### Necessary Libraries:
- [Scikit Learn](http://scikit-learn.org/stable/index.html)
    - Install `sklearn` latest python library
- [Pandas](http://pandas.pydata.org/)
    - Install `pandas` python library
- [Numpy](http://www.numpy.org/)
    - Install `numpy` python library
- [Matplotlib](https://matplotlib.org/)
    - Install `maplotlib` for python (version 3.7.3 recommended, pip install matplotlib==3.7.3)
- [Plotly](https://plot.ly/)
    - Install and signup for `plotly`
- [Seaborn](https://seaborn.pydata.org/)
    - Install and signup for `seaborn`
- [NLTK](http://www.nltk.org/)
    - Install `nltk` library
- [PAMI](https://github.com/UdayLab/PAMI?tab=readme-ov-file)
    - Install `PAMI` library
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
    - Install `UMAP` library
 
## Results
Model performance is evaluated using multiple metrics.
The best model, Multinomial Naive Bayes classifier, achieves high accuracy in classifying text into the 20 newsgroup categories.

## Future Improvements

(1) Experimenting with deep learning models (e.g., LSTM, BERT) for improved classification.

(2) Fine-tuning hyperparameters using grid search.

(3) Exploring topic modeling techniques like LDA.
