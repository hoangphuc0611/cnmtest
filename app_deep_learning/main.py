from flask import Flask, request, send_file
from flask_cors import CORS
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename
import os
import cv2
import pickle
import base64
import PIL.Image as Image
import io
import re

lb = pickle.loads(open('./lb.pkl', "rb").read())
model = keras.models.load_model('./resnet152.h5')
print('Model Loaded')

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, expose_headers='Authorization')

@app.route('/upload', methods=['POST'])
def predict():
    target = os.path.join(app.config['UPLOAD_FOLDER'], 'test')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']    
    print(file)
    filename = secure_filename(file.filename)
    destination = "/".join([target, filename])
    file.save(destination)
    return send_file('./rs.png', mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True)
