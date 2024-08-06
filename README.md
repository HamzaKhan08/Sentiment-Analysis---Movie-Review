## Sentiment Analysis of Movie Reviews by Hamza Khan
This project implements a sentiment analysis system that classifies movie reviews as positive or negative based on their text content. It utilizes both Naive Bayes and Logistic Regression models to perform the classification.

# Table of Contents
1. Project Overview
2. Dataset
3. Installation
4. Usage
5. Results
6. Improvements
7. Contributing
8. License

   
## Project Overview
The goal of this project is to demonstrate how to perform sentiment analysis on text data using machine learning models. It involves:

* Loading a dataset of movie reviews.
* Preprocessing the text (cleaning and tokenization).
* Converting the text into a numerical representation using Bag-of-Words and TF-IDF.
* Training Naive Bayes and Logistic Regression models.
* Evaluating model performance using accuracy and confusion matrices.
## Dataset
The dataset contains movie reviews with two columns:

text: The text of the movie review.
tag: The sentiment label ('pos' for positive and 'neg' for negative).
Ensure the dataset file (path_to_your_dataset.csv) is available in the project directory or specify the correct path.

## Installation
To run this project, you need Python and the following Python packages:

pandas
scikit-learn
nltk
You can install these packages using pip:
pip install pandas scikit-learn nltk

# Usage
Clone the repository: git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis

Ensure the dataset CSV file is in the project directory or update the path in the script.

Run the Python script: python sentiment_analysis.py


## Results
The initial results from running the script are as follows:

* Naive Bayes Model

Accuracy: 71%
Confusion Matrix: 
[[4469 1902]
 [1915 4658]]


* Logistic Regression Model

Accuracy: 69%
Confusion Matrix:
[[4452 1919]
 [2097 4476]]
These results indicate a solid baseline for sentiment classification.
