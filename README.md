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

## Conclusion:
The model is giving respectable accuracy but as we can see it is overfitting. To reduce the overfitting we can take the following measures:
1. Increase the training data.
2. Reduce the complexity of the model
3. Add a dropout layer.
4. Fine tune the model.

I could not work with the above solutions to make the val accuracy better due to shortage of time, but I plan on working on it eventually and try to increase the performance of the model.
