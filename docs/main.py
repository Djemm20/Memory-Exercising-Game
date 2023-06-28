# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# from pandas import *
#
# # load dataset
# data = load_iris()
#
# # store features and labels into appropriate variables
# X = data.data
# y = data.target
#
# # split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# dt = DecisionTreeClassifier()
#
# # train data
# dt.fit(X_train, y_train)
#
# # make prediction
# y_pred = dt.predict(X_test)
#
# print(X.shape)
# print(y.shape)
# print("Predictions:", y_pred)
# '''
# score = accuracy_score(y_test, y_pred)
#
# print("DecisionTree", score)
#
# plt.plot(y_test, y_pred)
#
# plt.xlabel = "Test Label"
#
# plt.ylabel = "Predicted Label"
#
# plt.title = "Decision Tree Classification Of Iris"
#
# plt.show()
# '''
import pickle
import numpy as np

def get_top_k_predictions(model, X_test, k):
    # get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)

    # GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:, -k:]

    # GET CATEGORY OF PREDICTIONS
    preds = [[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]

    preds = [item[::-1] for item in preds]

    return preds


model_path = "./model.pkl"
transformer_path = "./transformer.pkl"

loaded_model = pickle.load(open(model_path, 'rb'))
loaded_transformer = pickle.load(open(transformer_path, 'rb'))

test_features = loaded_transformer.transform(["Handled inbound customer inquiries, providing exceptional customer service and resolving issues in a timely manner."])
print(get_top_k_predictions(loaded_model, test_features, 10))
