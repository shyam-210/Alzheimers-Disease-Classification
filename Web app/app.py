from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from utils.Image_conversion import convert_image
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)


MODEL_PATH = "D:\Python\Projects\Alzheimer's Disease\Alzheimers-Disease-Classification\Web app\model\97_Non_Augmented.keras"
model = load_model(MODEL_PATH)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

UPLOAD_FOLDER = 'static/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = convert_image(filepath)

        preds = model.predict(img)
        preds = preds[0]

        class_names = ['Non-Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']

        percentages = [round(p * 100, 2) for p in preds]

        return render_template('results.html', image_file=file.filename, class_names=class_names, percentages=percentages, zip = zip)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
