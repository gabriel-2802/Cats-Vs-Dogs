from flask import Flask, request, render_template_string
import os
from werkzeug.utils import secure_filename
from AI.AIModel import AI_model  # Ensure this is your AI model class


def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize your AI model
    ai_model = AI_model()
    ai_model.set_optimal_theta('optimal_theta.npz')  # Set the path to your saved model

    with open('html/upload.html', 'r') as file:
        upload_html = file.read()

    with open('html/result.html', 'r') as file:
        result_html = file.read()

    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            if 'file' not in request.files:
                return 'No file part'
            file_req = request.files['file']
            if file_req.filename == '':
                return 'No selected file'
            if file_req:
                filename = secure_filename(file_req.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_req.save(file_path)
                prediction = ai_model.predict_image(file_path)  # Your AI model's prediction method
            return render_template_string(result_html, prediction=prediction)
        return render_template_string(upload_html)

    return app


def main():
    app = create_app()
    app.run(debug=True)


if __name__ == '__main__':
    main()
