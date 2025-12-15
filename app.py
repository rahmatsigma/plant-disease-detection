from flask import Flask, render_template, request, g
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import sqlite3
from datetime import datetime

# Import modul Grad-CAM
from gradcam import get_gradcam_heatmap, save_and_display_gradcam

app = Flask(__name__)

# --- KONFIGURASI ---
MODEL_PATH = 'models/best_model_v2.h5' # Pastikan pakai model terbaik Anda
CLASS_NAMES_PATH = 'models/class_names.txt'
UPLOAD_FOLDER = 'static/uploads'
DATABASE = 'plant_data.db'
LAST_CONV_LAYER_NAME = "top_activation" 

# Logic: Batas minimal kepercayaan AI. Jika di bawah ini, anggap "Tidak Dikenali"
CONFIDENCE_THRESHOLD = 50.0 

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- FUNGSI DATABASE HELPER ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row # Agar bisa akses kolom pakai nama (row['description'])
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# --- LOAD MODEL & CLASS NAMES ---
print("Memuat model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded.")
except:
    print(f"Model {MODEL_PATH} tidak ditemukan. Pastikan sudah training.")
    model = None

def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            return f.read().splitlines()
    return []

class_names = load_class_names()

def predict_image_process(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    predicted_class = class_names[pred_index]
    
    return predicted_class, confidence, img_array, pred_index

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_url = None
    gradcam_url = None 
    db_info = None
    warning_msg = None

    if request.method == 'POST':
        if 'file' not in request.files: return "Tidak ada file"
        file = request.files['file']
        if file.filename == '': return "Nama file kosong"

        if file:
            # Tambahkan timestamp di nama file agar unik dan tidak tertimpa
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # 1. Prediksi
                pred_class, conf, img_array, pred_idx = predict_image_process(filepath)
                
                img_url = filepath
                confidence = round(conf, 2)

                # 2. LOGIC: Cek Threshold Confidence
                if confidence < CONFIDENCE_THRESHOLD:
                    prediction = "Tidak Dikenali / Tidak Yakin"
                    warning_msg = "Tingkat kepercayaan AI terlalu rendah. Kemungkinan bukan daun kentang atau gambar buram."
                else:
                    prediction = pred_class
                    
                    # 3. DATABASE: Ambil Info Penyakit
                    cur = get_db().cursor()
                    cur.execute("SELECT * FROM diseases WHERE class_name = ?", (prediction,))
                    db_info = cur.fetchone()

                    # 4. DATABASE: Simpan Riwayat (History Log)
                    cur.execute("INSERT INTO scan_history (filename, prediction, confidence) VALUES (?, ?, ?)",
                                (filename, prediction, confidence))
                    get_db().commit()

                    # 5. Grad-CAM (Hanya jika yakin)
                    try:
                        heatmap = get_gradcam_heatmap(model, img_array, LAST_CONV_LAYER_NAME, pred_idx)
                        gradcam_filename = "gradcam_" + filename
                        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                        save_and_display_gradcam(filepath, heatmap, gradcam_path)
                        gradcam_url = gradcam_path
                    except Exception as e:
                        print(f"Grad-CAM Error: {e}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"Error: {str(e)}"

    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence, 
                           img_url=img_url,
                           gradcam_url=gradcam_url,
                           info=db_info,
                           warning=warning_msg)

# --- FITUR BARU: HALAMAN RIWAYAT ---
@app.route('/history')
def history():
    cur = get_db().cursor()
    # Ambil 20 riwayat terakhir, urutkan dari yang terbaru
    cur.execute("SELECT * FROM scan_history ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    return render_template('history.html', rows=rows)

if __name__ == '__main__':
    app.run(debug=True)