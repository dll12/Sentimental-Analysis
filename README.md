# Sentimental-Analysis

Project Title: Sentimental Analysis of tweets

Description: There are millions of tweets posted on twitter everyday. This project attempts to predict the sentiment of the tweets into 7 Categories that are: Anger,Sadness,worry,neutral,surprise,fun,happiness.

Dataset: Dataset was fetched from Kaggle.The dataset includes 4 columns i.e. tweet_id,sentiment,autor,content.
We extracted the sentiment and content from the dataset for further processing.

Project Detail: During Preprocessing, The dataset was containing some empty emotions and some non specific emotions which were deleted from dataset. Then the dataset was divided into training and testing set with ratio of 80%-20%.
Then since the sentiments were string so we needed to convert/encode them so that we can use machine learnin models on them.Then we tokenized the tweets and removed stop words,hashtags,links from it.
We used Bag of Words(BOW) model for this project.Then we represented every tweet with the conversion from BOW model to a feature.
Then we used Artificial Neural Network Model and trained the model with the given training data.

Outcome: We got an accuracy score of 38.68 with 7 class classification.

Further Improvements: We can improve the accuracy be replacing the BOW model by word vectorisation and replacing ANN with Recurrent Neural Networks. 
