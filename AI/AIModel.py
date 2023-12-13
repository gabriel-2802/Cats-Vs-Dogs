import os

from PhotoEval import *


class AI_model:
    def __init__(self):
        self.optimal_theta = None
        self.k = 40
        self.epsilon = 1e-2
        self.max_steps = 2000
        self.lmb = 40

    def train(self, data):
        cols_x = np.size(data['X'], 1)
        theta1 = initialize_weights(self.epsilon, self.k, cols_x + 1)
        theta2 = initialize_weights(self.epsilon, 1, self.k + 1)
        initial_theta = np.concatenate((theta1.flatten(), theta2.flatten()))
        cf = lambda theta: cost_function(theta, data['X'], data['y'], self.lmb)
        optimal_theta, cost, num_iterations = fmincg(cf, initial_theta,
                                                     {'MaxIter': self.max_steps})

        self.optimal_theta = optimal_theta

    def accuracy(self, data, cv_data):
        pred = forward_progation(data['X'], self.optimal_theta)
        pred = np.round(pred)
        accuracy = 1 - np.mean(np.abs(pred - data['y']))
        print("Using our custom AI model: ")
        print("Accuracy on training set: ", accuracy)

        pred_cv = forward_progation(cv_data['X'], self.optimal_theta)
        pred_cv = np.round(pred_cv)
        accuracy_cv = 1 - np.mean(np.abs(pred_cv - cv_data['y']))
        print("Accuracy on CV set: ", accuracy_cv)
        print("\n")

    def save_optimal_theta(self):
        np.savez("optimal_theta.npz", optimal_theta=self.optimal_theta)

    def predict_image(self, path):
        x = load_image(path)
        x = x.reshape(1, -1)

        # load the pretrained parameters
        optimal_theta = np.load("optimal_theta.npz")['optimal_theta']

        prediction = np.round(forward_progation(x, optimal_theta))

        if prediction == 0:
            return "Is a cat"
        else:
            return "Is a dog"

    def predict_photo(self, path):
        print("Predicting photos in the directory: ", path)
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            predict_photo(full_path)

    def set_optimal_theta(self, path):
        self.optimal_theta = np.load(path)['optimal_theta']
