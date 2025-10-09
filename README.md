# Predicting Power Outages

## What is this?
The aim of this project is to create models, and tie them together in an ensemble, to predict whether a power outage will occur in New Zealand based on weather data.


## How does it work?
The main script is...
What does our code do?
First, we have preprocessed our weather data, normalising it, and making sure everything is in a consistent format. After this, we paired it with our outage dataset to generate entries that detail the weather for a given time, and a binary flag for whether or not a power outage occured on that day. With this, we can begin our training. We trained the following models:
- K-NN
- XGBoost
- Random Forests
- LSTM
- SVM
To utilise the strengths of each model, we assembled an ensemble in a stacking manner, and predict using the majority vote. We focus on the recall of our model, i.e., what percentage of "Yes, power outage occured" are we predicting as "Yes, power outage occured". 


## Authors
- Alex Timpany
- Ethan Jackson
- Yinchi Tan
- Zitao Wang
- Yisai Zhang
- Yusin Zhang
