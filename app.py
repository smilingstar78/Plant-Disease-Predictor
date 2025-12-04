from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model('plant_disease_model.h5')

# Class mapping
idx_to_class = {
    0: 'Apple Scab', 1: 'Apple Black Rot', 2: 'Apple Cedar Apple Rust',
    3: 'Apple Healthy', 4: 'Blueberry Healthy', 5: 'Cherry Powdery Mildew',
    6: 'Cherry Healthy', 7: 'Corn Gray Leaf Spot', 8: 'Corn Common Rust',
    9: 'Corn Northern Leaf Blight', 10: 'Corn Healthy', 11: 'Grape Black Rot',
    12: 'Grape Esca', 13: 'Grape Leaf Blight', 14: 'Grape Healthy',
    15: 'Orange Citrus Greening', 16: 'Peach Bacterial Spot', 17: 'Peach Healthy',
    18: 'Pepper Bacterial Spot', 19: 'Pepper Healthy', 20: 'Potato Early Blight',
    21: 'Potato Late Blight', 22: 'Potato Healthy', 23: 'Raspberry Healthy',
    24: 'Soybean Healthy', 25: 'Squash Powdery Mildew', 26: 'Strawberry Leaf Scorch',
    27: 'Strawberry Healthy', 28: 'Tomato Bacterial Spot', 29: 'Tomato Early Blight',
    30: 'Tomato Late Blight', 31: 'Tomato Leaf Mold', 32: 'Tomato Septoria Leaf Spot',
    33: 'Tomato Spider Mites', 34: 'Tomato Target Spot', 35: 'Tomato Yellow Leaf Curl Virus',
    36: 'Tomato Mosaic Virus', 37: 'Tomato Healthy'
}

# Home Page
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        img_file = request.files['file']
        img_path = os.path.join('static', img_file.filename)
        img_file.save(img_path)

        img = image.load_img(img_path, target_size=(160,160))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        predicted_index = np.argmax(pred, axis=1)[0]
        prediction = idx_to_class[predicted_index]

    return render_template('index.html', prediction=prediction)

# About Project Page
@app.route('/project')
def project():
    return render_template('project.html')

# About Me Page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
