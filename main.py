from flask import Flask, request, render_template_string, redirect, url_for
import os
from werkzeug.utils import secure_filename
from AI.AIModel import AI_model  # Ensure this is your AI model class

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize your AI model
ai_model = AI_model()
ai_model.set_optimal_theta('optimal_theta.npz')  # Set the path to your saved model

# HTML Templates as Strings
upload_html = '''
<!doctype html>
<html>
  <head>
    <title>AI predictor</title>
  </head>
  <body>
    <h1>Predict a photo</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
</form>
  </body>
</html>
'''

result_html = '''
<!doctype html>
<html>
  <head>
    <title>Result</title>
  </head>
  <body>
    <h1>Prediction Result</h1>
    <p>{{ prediction }}</p>
    <a href="/">Try another image</a>
  </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = ai_model.predict_image(file_path)  # Your AI model's prediction method
        return render_template_string(result_html, prediction=prediction)
    return render_template_string(upload_html)

if __name__ == '__main__':
    app.run(debug=True)