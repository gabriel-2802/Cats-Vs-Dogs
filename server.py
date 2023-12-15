import numpy as np
from flask import Flask, request, render_template_string
import os

from sklearn.linear_model import LogisticRegression
from werkzeug.utils import secure_filename
import AI.AiModel as ai


def demo():
    data = np.load("input_data/train_data.npz")
    cv_data = np.load("input_data/CV_data.npz")

    print("Using an AI model from the library sklearn: ")
    model = LogisticRegression(max_iter=1000)
    model.fit(data['X'], data['y'])
    print("Accuracy on training set: ", model.score(data['X'], data['y']))
    print("Accuracy on CV set: ", model.score(cv_data['X'], cv_data['y']))
    print("\n")

    print("Using our own AI model: ")
    ai_model = ai.AiModel()
    # ai_model.train(data)
    ai_model.set_optimal_theta("optimal_theta.npz")

    print("Accuracy on training set: ", ai_model.accuracy(data))
    print("Accuracy on CV set: ", ai_model.accuracy(cv_data))
    print("\n")


def create_app():
    # Initialize Flask app
    app = Flask(__name__)

    # Set the folder where uploaded files will be stored
    app.config['UPLOAD_FOLDER'] = 'uploads/'

    # Create the upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize the AI model and set its parameters
    ai_model = ai.AiModel()
    ai_model.set_optimal_theta('optimal_theta.npz')

    # Load the HTML templates for upload and result pages
    with open('html/upload.html', 'r') as file:
        upload_html = file.read()
    with open('html/result.html', 'r') as file:
        result_html = file.read()

    # Define route for file upload
    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        # Handle file upload via POST request
        if request.method == 'POST':
            # Check if the file part is present in the request
            if 'file' not in request.files:
                return 'No file part'

            # Get the file from the request
            file_req = request.files['file']

            # Check if a file was selected
            if file_req.filename == '':
                return 'No selected file'

            # Save the file and make a prediction
            if file_req:
                # Secure the filename and create a full path
                filename = secure_filename(file_req.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Save the uploaded file
                file_req.save(file_path)

                # Use the AI model to make a prediction based on the uploaded file
                prediction = ai_model.predict_image(file_path)
                return render_template_string(result_html, prediction=prediction)

        # Render the upload page for GET request
        return render_template_string(upload_html)

    # Return the configured Flask app
    return app


def main():
    app = create_app()
    app.run(debug=True)


if __name__ == '__main__':
    main()
