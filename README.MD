# Email Spam Classification

This repository contains code for training a spam classification model using the Naive Bayes algorithm. It also includes functions for evaluating the model's performance and visualizing the spamicity of a given file.

## Prerequisites
- Python 3.x
- NLTK library
- Matplotlib library
- NumPy library

## Installation
1. Clone the repository: `git clone https://github.com/your-username/your-repository.git`
2. Install the required dependencies: `pip install nltk matplotlib numpy`
3. Install nltk stop words: `import nltk nltk.download('stopwords')`

## Usage
1. Import the necessary modules:
```python
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
import numpy as np
import re
```

2. Train the spam classification model by calling the `train_model` function:
```python
train_model(training_percent=0.8, SPAM_FOLDER='HAMS', HAM_FOLDER='SPAMS')
```
This function will randomly select a percentage of files from the provided spam and ham folders for training the model. It will store the training and testing file lists in separate text files.

3. Classify a file's spamicity using the `get_file_spamicity` function:
```python
spamicity = get_file_spamicity(filename, n=8, plot=False)
```
This function calculates the spamicity of a given file by comparing the words in the file to the trained word count dictionary. It returns the calculated spamicity value.

If plot=True you will get this plot :
![alt text](https://github.com/An0n1mity/SpamClassifierEval/blob/master/get_file_spamicity_plot.png)

4. Test misclassification for a given `n` using the `test_misclassification` function:
```python
test_misclassification(testing_files_spams, testing_files_hams, n=(8, 16, 32), threshold=0.6, unseen_spamicity=0.4, plot=False, verbose=False)
```
This function tests the misclassification rate of the spam classification model on the provided testing files. It compares the calculated spamicity of each file to a threshold value and counts the false positives and true negatives. It accepts an optional `n` parameter to specify the number of words used for classification. 

If plot=True you will get this plot :
![alt text](https://github.com/An0n1mity/SpamClassifierEval/blob/master/test_misclassification_plot.png)
