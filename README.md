# Sentiment Analysis for Customer Reviews

## Overview
This project implements a sentiment analysis model for classifying customer reviews into positive, negative, or neutral sentiments. It utilizes a Long Short-Term Memory (LSTM) neural network for sequence processing and is trained on a dataset of customer reviews from an online fashion retailer.

## Features
<ul><li>Preprocessing of textual data using tokenization and padding techniques.</li>
<li>LSTM-based neural network architecture for sentiment analysis.</li>
<li>Visualization of accuracy and loss graphs during training.</li>
<li>Word cloud visualization for positive, neutral, and negative reviews.</li>
<li>Exploratory Data Analysis (EDA) techniques for understanding the dataset.</li></ul>

## Prerequisites
<ul><li>Python 3.x</li>
<li>Required Python packages: pandas, numpy, tensorflow, scikit-learn, matplotlib, seaborn, wordcloud</li>

## File Structure
<ul><li>sentiment_analysis.py: Main Python script for sentiment analysis, data preprocessing, model training, and visualization.</li>
<li>Womens Clothing E-Commerce Reviews.csv: Sample dataset containing customer reviews and corresponding ratings. The data was taken from Kaggle-https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews</li>
<li>README.md: Project README file.</li></ul>

## Nature and Characteristics of the Dataset
The dataset used in this project consists of customer reviews from an online fashion retailer. It includes 23486 rows and 10 feature variables:
1. Clothing ID: Integer categorical variable indicating the specific product being reviewed.
2. Age: Positive integer variable representing the reviewer's age.
3. Title: String variable for the review title.
4. Review Text: String variable for the review body.
5. Rating: Positive ordinal integer variable indicating the product score from 1 (worst) to 5 (best).
6. Recommended IND: Binary variable indicating whether the customer recommends the product (1 for recommended, 0 for not recommended).
7. Positive Feedback Count: Positive integer indicating the number of other customers who found the review positive.
8. Division Name: Categorical name of the product high-level division.
9. Department Name: Categorical name of the product department.
10. Class Name: Categorical name of the product class.

## Data Preprocessing Steps
1. Handling Missing Values: Rows with missing values in the 'Review Text' column were removed to ensure data quality.
2. Text Tokenization: Review text was tokenized into individual words to prepare it for input into the neural network.
3. Padding: Tokenized sequences were padded to ensure uniform length for input into the LSTM network.

## Feature Engineering Techniques Applied
"Sentiment" feature was created from "Rating". It is used as our target variable to train the mode. Where rating is 4 or 5, sentiment is positive (2), for rating 3 sentiment is neutral (1), and for rating 1 and 2 sentiment is 0 (negative). 

## Model Architecture and Development Process
1. Embedding Layer: Used to convert words into dense vectors of fixed size.
2. LSTM Layer: Implemented a Long Short-Term Memory network to capture sequential patterns in the reviews.
3. Dense Layer: Output layer with softmax activation for multi-class classification.
The development process involved experimenting with various LSTM architectures, tuning hyperparameters, and optimizing the model for the best validation performance.

## Training Details, Including Hyperparameters
1. Epochs: 10
2. Batch Size: 64
3. Optimizer: Adam
4. Learning Rate: 0.001
5. Loss Function: Sparse categorical cross-entropy

## Evaluation Metrics and Results
1. Training Accuracy: Achieved approximately 80% accuracy on the training set.
2. Validation Accuracy: Validation accuracy plateaued at around 75% after 10 epochs.
3. Loss: Training loss decreased, but validation loss plateaued and started increasing, indicating overfitting.

## Visualizations and Their Interpretations
Accuracy and Loss Graphs: Plots showed training accuracy increasing and training loss decreasing, but validation accuracy reached a plateau and validation loss started increasing after a few epochs, indicating overfitting.

## Integration into an Application
A pickle file of the model has been created. To integrate the model into an application, expose it as an API endpoint using a web framework like Flask or FastAPI. Provide the review text as input to the endpoint, and the model will return the predicted sentiment (positive, negative, or neutral). 

## Conclusion:
The model is giving respectable accuracy (78.85%) but as we can see it is overfitting. To reduce the overfitting we can take the following measures:
1. Increase the training data.
2. Reduce the complexity of the model
3. Add a dropout layer.
4. Fine tune the model.

I could not work with the above solutions to make the val accuracy better due to shortage of time, but I plan on working on it eventually and try to increase the performance of the model.
