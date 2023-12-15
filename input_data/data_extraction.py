import os
import numpy as np
from PIL import Image


def load_channel(image):
    # Calculate the size of the image
    m, n = image.shape
    length = 256

    # Initialize a zero array for the histogram
    channel = np.zeros(length)

    # Count the frequency of each pixel value
    for i in range(length):
        channel[i] = np.sum(image == i)

    # Normalize the histogram
    channel /= (m * n / length)

    return channel


def load_image(path):
    row = col = 300

    img = Image.open(path)
    # Use Image.Resampling.LANCZOS for high-quality downsampling
    img = img.resize((row, col), Image.Resampling.LANCZOS)

    # Convert image to a numpy array and round pixel values
    img = np.array(img, dtype=float)
    img = np.round(img)

    # Since all images are RGB, process each of the three channels
    channels = [load_channel(img[:, :, i]) for i in range(3)]

    # Concatenate the channels and return the feature vector as a column vector
    x = np.concatenate(channels).T

    return x

def get_data(dir_path):
    file_list = os.listdir(dir_path)

    X = []
    y = []

    for file in file_list:
        filepath = os.path.join(dir_path, file)

        # Skip directories
        if os.path.isdir(filepath):
            continue

        # Load image and append it to the dataset
        img = load_image(filepath)
        X.append(img)

        # Append label based on the filename
        if "cat" in file:
            y.append(0)
        elif "dog" in file:
            y.append(1)

    return np.array(X), np.array(y)


def get_train_data():
    cat_path = "../../data/train/cats"
    dog_path = "../../data/train/dogs"

    # Initialize feature matrix and label vector
    X = []
    y = []

    # Load and append data from the cat and dog directories
    cat_X, cat_y = get_data(cat_path)
    dog_X, dog_y = get_data(dog_path)

    # Append the data to the feature matrix and label vector
    X.extend(cat_X)
    X.extend(dog_X)
    y.extend(cat_y)
    y.extend(dog_y)

    # Convert the data to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Shuffle the data
    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y = y[permutation]

    np.savez("train_data.npz", X=X, y=y)


def get_CV_data():
    cat_path = "../../data/train/train_cv/cats"
    dog_path = "../../data/train/train_cv/dogs"

    cat_X, cat_y = get_data(cat_path)
    dog_X, dog_y = get_data(dog_path)

    X = np.concatenate((cat_X, dog_X))
    y = np.concatenate((cat_y, dog_y))

    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y = y[permutation]

    np.savez("CV_data.npz", X=X, y=y)
