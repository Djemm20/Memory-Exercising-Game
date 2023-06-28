# Import necessary libraries
import os
import tweepy
from pyexpat import features
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

# Initialize Tweepy
api_key = 'WC2RfdKkc3I3jvqW5Fk39ZSLJ'
api_key_secret = 'perfdWp84L3tqcmCkWqKlP4jy3xXIrZtTLiWNgPESy9wPQp4ub'
access_token = '2459334069-dYcyulBBJC7YZOnlW8SihUWSdWl40SnSJxbdxD8'
access_token_secret = 'HLFh68ySEJVNSjVoQEO7bsGBTBBEfcvxOsxX4Qmh53lOm'

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Data collection function


def collect_data(usernames):
    data = []
    # Implement the logic to collect data from profiles and tweets of given usernames
    for username in usernames:
        profile = api.get_user(screen_name=username)
        tweets = api.user_timeline(screen_name=username, count=100)

        profile_data = [username, profile.followers_count, profile.friends_count, profile.statuses_count]

        tweet_data = [tweet.text for tweet in tweets]

        # Appending the collected data to the list
        data.append([profile_data, tweet_data])

    # Return as a DataFrame
    return pd.DataFrame(data)

# Data preprocessing and feature extraction function


def preprocess_and_extract_features(data):
    # Implement the logic to preprocess data and extract features
    '''
    # Preprocess data
    data['tweets'] = data['tweets'].str.replace('http\S+|www.\S+', '', regex=True)
    data['tweets'] = data['tweets'].str.replace('[^\w\s]', '')

    # Perform feature extraction
    data['word_count'] = data['tweets'].apply(lambda x: len(x.split()))
    data['average_word_length'] = data['tweets'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))
    '''
    # Select features for building the classifier
    features = data['followers_count', 'friends_count', 'statuses_count']

    # Select target to be predicted
    labels = data['label']


    # Return as a DataFrame with features
    return features, labels

# Model building and evaluation function


def build_and_evaluate_model(features, labels):
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

    # Train the decision tree model
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_features, train_labels)

    # Predict on the test set
    predictions = clf.predict(test_features)

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    # Print the results
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Visualization function


def visualize_tree(clf):
    # Implement the logic to visualize the decision tree

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), dpi=300)
    tree.plot_tree(clf, feature_names=features.columns, class_names=["Negative", "Positive"], filled=True)
    plt.xlabel = "Test Label"
    plt.ylabel = "Predicted Label"
    plt.title = "Decision Tree Classification Of Tweets"
    plt.show()


# Main function


def main():
    # Collect data
    usernames = ['username1', 'username2', ...]  # The usernames of the profiles to be collected
    data = collect_data(usernames)

    # Preprocess data and extract features
    features, labels = preprocess_and_extract_features(data)

    # Build and evaluate model
    build_and_evaluate_model(features, labels)


if __name__ == "__main__":
    main()
