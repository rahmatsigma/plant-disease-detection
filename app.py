from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

from gradcam import get_gradcam_heatmap, save_and_display_gradcam

app = Flask(__name__)

# --- KONFIGURASI ---
MODEL_PATH = 'models/plant_disease_model.h5'
CLASS_NAMES_PATH = 'models/class_names.txt'
UPLOAD_FOLDER = 'static/uploads'
# --- TAMBAHAN GRAD-CAM: Nama layer target (Ganti jika hasil cek_model.py berbeda) ---
LAST_CONV_LAYER_NAME = "out_relu" 

# Pastikan folder upload ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- LOAD MODEL ---
print("Memuat model... Tunggu sebentar.")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model berhasil dimuat!")

def load_class_names():
    with open(CLASS_NAMES_PATH, 'r') as f:
        return f.read().splitlines()

class_names = load_class_names()

# Fungsi ini dimodifikasi untuk mengembalikan juga index kelas prediksi
def predict_image_process(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions) # Dapatkan index kelas tertinggi
    predicted_class = class_names[pred_index]
    confidence = np.max(predictions) * 100
    
    # --- TAMBAHAN GRAD-CAM: Kembalikan img_array dan pred_index ---
    return predicted_class, confidence, img_array, pred_index

# --- ROUTE WEBSITE ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_url = None
    gradcam_url = None # Variabel baru untuk URL gambar Grad-CAM

    if request.method == 'POST':
        if 'file' not in request.files: return "Tidak ada file"
        file = request.files['file']
        if file.filename == '': return "Nama file kosong"

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Panggil fungsi prediksi yang sudah diupdate
                prediction, confidence, img_array, pred_index = predict_image_process(filepath)
                img_url = filepath 

                # --- TAMBAHAN GRAD-CAM: Proses Pembuatan Heatmap ---
                print("Membuat Grad-CAM...")
                # 1. Hitung Heatmap
                heatmap = get_gradcam_heatmap(model, img_array, LAST_CONV_LAYER_NAME, pred_index)
                
                # 2. Tentukan nama file output untuk Grad-CAM
                gradcam_filename = "gradcam_" + filename
                gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                
                # 3. Gabungkan dan simpan gambar
                save_and_display_gradcam(filepath, heatmap, gradcam_path)
                
                # 4. Set URL untuk ditampilkan di HTML
                gradcam_url = gradcam_path
                print("Grad-CAM selesai.")

            except Exception as e:
                # Print error lengkap ke terminal untuk debugging
                import traceback
                traceback.print_exc()
                return f"Terjadi error: {str(e)}. Cek terminal."

    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence, 
                           img_url=img_url,
                           gradcam_url=gradcam_url) # Kirim URL baru ke HTML

if __name__ == '__main__':
    app.run(debug=True)