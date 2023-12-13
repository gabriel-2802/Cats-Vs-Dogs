from sklearn.linear_model import LogisticRegression

from PhotoEval import *

def create_model():
    data = np.load("DataExtract/train_data.npz")
    cv_data = np.load("DataExtract/CV_data.npz")

    rows_x = np.size(data['X'], 0)
    cols_x = np.size(data['X'], 1)
    rows_cv = np.size(cv_data['X'], 0)
    cols_cv = np.size(cv_data['X'], 1)

    #  k = neurons in hidden layer
    k = 40
    # epsilon - chosen to initialise weights
    epsilon = 1e-2

    theta1 = initialize_weights(epsilon, k, cols_x + 1)
    theta2 = initialize_weights(epsilon, 1, k + 1)

    initial_theta = np.concatenate((theta1.flatten(), theta2.flatten()))
    # we have over 30k parameters in our network, therefore we need a large number of steps
    max_steps = 2000
    lmb = 40

    # create a lambda function to pass to the minimization function
    cf = lambda theta: cost_function(theta, data['X'], data['y'], lmb)
    optimal_theta, cost, num_iterations = fmincg(cf, initial_theta, {'MaxIter': max_steps})

    pred = forward_progation(data['X'], optimal_theta)
    pred = np.round(pred)
    accuracy = 1 - np.mean(np.abs(pred - data['y']))
    print("Using our custom AI model: ")
    print("Accuracy on training set: ", accuracy)

    pred_cv = forward_progation(cv_data['X'], optimal_theta)
    pred_cv = np.round(pred_cv)
    accuracy_cv = 1 - np.mean(np.abs(pred_cv - cv_data['y']))
    print("Accuracy on CV set: ", accuracy_cv)
    print("\n")

    np.savez("optimal_theta.npz", optimal_theta=optimal_theta)

    print("Using an AI model from the library sklearn: ")
    model = LogisticRegression(max_iter=1000)
    model.fit(data['X'], data['y'])
    print("Accuracy on training set: ", model.score(data['X'], data['y']))
    print("Accuracy on CV set: ", model.score(cv_data['X'], cv_data['y']))
    print("\n")

import os
from PhotoEval import predict_photo
def demo():
    path = "ourTest"

    print("Predicting photos in the directory: ", path)
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        predict_photo(full_path)

if __name__ == "__main__":
    create_model()
    demo()
