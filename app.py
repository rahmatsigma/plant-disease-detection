from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# --- KONFIGURASI ---
MODEL_PATH = 'models/plant_disease_model.h5'
CLASS_NAMES_PATH = 'models/class_names.txt'
UPLOAD_FOLDER = 'static/uploads'

# Pastikan folder upload ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- LOAD MODEL (Hanya sekali saat server nyala) ---
print("Memuat model... Tunggu sebentar.")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model berhasil dimuat!")

def load_class_names():
    with open(CLASS_NAMES_PATH, 'r') as f:
        return f.read().splitlines()

class_names = load_class_names()

def predict_image_process(img_path):
    # Preprocessing sama persis dengan training
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    return predicted_class, confidence

# --- ROUTE WEBSITE ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_url = None

    if request.method == 'POST':
        # Cek apakah ada file yang diupload
        if 'file' not in request.files:
            return "Tidak ada file yang diupload"
        
        file = request.files['file']
        
        if file.filename == '':
            return "Nama file kosong"

        if file:
            # Simpan file sementara
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Lakukan prediksi
            try:
                prediction, confidence = predict_image_process(filepath)
                # Path untuk ditampilkan di HTML
                img_url = filepath 
            except Exception as e:
                return f"Terjadi error saat prediksi: {str(e)}"

    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence, 
                           img_url=img_url)

if __name__ == '__main__':
    # Debug=True artinya kalau ada error code berubah, server auto-restart
    app.run(debug=True)