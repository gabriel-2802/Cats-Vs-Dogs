from processing import *
from abc import ABC, abstractmethod

class Ai(ABC):

    def __init__(self):
        self.k = 40
    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def accuracy(self, data):
        pass

    @abstractmethod
    def predict_image(self, path):
        pass



class AiModel(Ai):
    def __init__(self):
        super().__init__()
        self.optimal_theta = None
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

    def accuracy(self, data):
        pred = forward_progation(data['X'], self.optimal_theta)
        pred = np.round(pred)
        accuracy = 1 - np.mean(np.abs(pred - data['y']))
        return accuracy

    def save_optimal_theta(self, filename):
        np.savez(filename, optimal_theta=self.optimal_theta)

    def predict_image(self, path):
        x = load_image(path)
        x = x.reshape(1, -1)

        # load the pretrained parameters
        if self.optimal_theta is None:
            self.optimal_theta = np.load("optimal_theta.npz")['optimal_theta']

        prediction = np.round(forward_progation(x, self.optimal_theta))

        if prediction == 0:
            return "The given photo is likely a cat."
        else:
            return "The given photo is likely a dog."

    def set_optimal_theta(self, path):
        self.optimal_theta = np.load(path)['optimal_theta']
