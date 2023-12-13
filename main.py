from sklearn.linear_model import LogisticRegression
import os

from AI.AIModel import AI_model
from PhotoEval import *

def official_AI_model(data, cv_data):
    print("Using an AI model from the library sklearn: ")
    model = LogisticRegression(max_iter=1000)
    model.fit(data['X'], data['y'])
    print("Accuracy on training set: ", model.score(data['X'], data['y']))
    print("Accuracy on CV set: ", model.score(cv_data['X'], cv_data['y']))
    print("\n")

def main():
    AI = AI_model()
    data = np.load("DataExtract/train_data.npz")
    cv_data = np.load("DataExtract/CV_data.npz")
    AI.train(data)
    AI.accuracy(data, cv_data)

    official_AI_model(data, cv_data)

    AI.save_optimal_theta()

    # You can add the path to your own directory here
    PATH = "ourTest"
    AI.predict_photo(PATH)

if __name__ == "__main__":
    main()
