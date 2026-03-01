# Urgency Sentiment Analysis
Using machine learning to detect phishing emails based on urgency-based language patterns.

## Overview
This project explores the correlation between urgency-based language in emails and phishing probability. Built as my General Beadle Honors Thesis at Dakota State University, this project applies sentiment analysis and NLP techniques to phishing datasets to identify predictive linguistic patterns.

## Approach
Data: Phishing dataset (dataset.csv with Text and Score columns)

Preprocessing: Text cleaning (URLs, mentions, numbers removed), tokenization via BERT tokenizer

Model: BERT (bert-base-uncased) with custom regression head for urgency scoring

Key Features: Urgency indicators via sentiment analysis, linguistic patterns

## Results
Model achieves >99% accuracy on test dataset

The higher density of phishing within a database indicates a higher likelihood of high urgency messages.

## Tech Stack
Python

TensorFlow

Transformers (BERT)

scikit-learn

Pandas, NumPy

## Running the Project
### Install dependencies
`pip install tensorflow transformers pandas scikit-learn`

### Run analysis
`python urgency_sentiment_v1.py`

## Files
urgency_sentiment_v1.py — Main model code

modeltesting3.py — Additional testing

dataset.csv — Phishing dataset (not included, add your own)

## Links
Thesis Paper: https://scholar.dsu.edu/cgi/viewcontent.cgi?article=1008&context=honors

LinkedIn: https://www.linkedin.com/in/patrick-rau/

---

Built as part of my BS in Cyber Operations at Dakota State University. Defended May 2025.
