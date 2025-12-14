from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

# Import modul Grad-CAM Anda
from gradcam import get_gradcam_heatmap, save_and_display_gradcam

app = Flask(__name__)

# --- KONFIGURASI ---
# Gunakan model baru hasil training EfficientNet
MODEL_PATH = 'models/best_model_v2.h5' 
CLASS_NAMES_PATH = 'models/class_names.txt'
UPLOAD_FOLDER = 'static/uploads'

# --- KONFIGURASI GRAD-CAM EFFICIENTNET ---
# Untuk EfficientNetB0, layer konvolusi terakhir biasanya bernama "top_activation"
# Jika error "layer not found", gunakan cek_model.py untuk melihat nama pastinya.
LAST_CONV_LAYER_NAME = "top_activation"

# Pastikan folder upload ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- DATABASE SOLUSI (FITUR KOMPLEKSITAS) ---
# Ini membuat aplikasi tidak hanya mendeteksi, tapi memberi solusi.
solutions_db = {
    "Potato___Early_blight": {
        "description": "Bercak Kering (Early Blight) disebabkan oleh jamur Alternaria solani. Gejalanya berupa bercak cincin konsentris pada daun tua.",
        "treatment": "1. Pangkas daun yang terinfeksi.\n2. Gunakan fungisida berbahan aktif Klorotalonil atau Tembaga.\n3. Beri jarak tanam agar sirkulasi udara baik."
    },
    "Potato___Late_blight": {
        "description": "Busuk Daun (Late Blight) adalah penyakit paling mematikan pada kentang, disebabkan oleh Phytophthora infestans.",
        "treatment": "1. Segera musnahkan tanaman yang terinfeksi berat.\n2. Gunakan fungisida sistemik (misal: Metalaksil).\n3. Hindari penyiraman sore hari agar daun tidak lembab malam hari."
    },
    "Potato___healthy": {
        "description": "Tanaman Anda terlihat sehat dan subur!",
        "treatment": "1. Lanjutkan pemupukan berimbang (NPK).\n2. Pantau hama secara berkala.\n3. Jaga kelembaban tanah tetap stabil."
    }
}

# --- LOAD MODEL ---
print("Memuat model EfficientNet... Tunggu sebentar.")
try:
    model = load_model(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model: {e}")
    print("Pastikan Anda sudah menjalankan train.py dan file models/best_model_v2.h5 ada.")

def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            return f.read().splitlines()
    else:
        return ["Kelas 1", "Kelas 2", "Kelas 3"] # Fallback jika file tidak ada

class_names = load_class_names()

def predict_image_process(img_path):
    # Load gambar sesuai ukuran input EfficientNet (224x224)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocessing: EfficientNet biasanya bekerja baik dengan 0-1 (rescale 1./255)
    # Ini harus SAMA PERSIS dengan konfigurasi di train.py (ImageDataGenerator)
    img_array /= 255.0

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions) # Index probabilitas tertinggi
    predicted_class = class_names[pred_index]
    confidence = np.max(predictions) * 100
    
    return predicted_class, confidence, img_array, pred_index

# --- ROUTE WEBSITE ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_url = None
    gradcam_url = None 
    solution_info = None # Variabel untuk menyimpan info solusi

    if request.method == 'POST':
        if 'file' not in request.files: return "Tidak ada file"
        file = request.files['file']
        if file.filename == '': return "Nama file kosong"

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # 1. Lakukan Prediksi
                prediction, confidence, img_array, pred_index = predict_image_process(filepath)
                img_url = filepath 

                # 2. Ambil Solusi dari Database
                # Menggunakan .get() agar tidak error jika kelas tidak dikenali
                solution_info = solutions_db.get(prediction, {
                    "description": "Keterangan tidak tersedia.",
                    "treatment": "Hubungi ahli pertanian terdekat."
                })

                # 3. Buat Grad-CAM
                print(f"Membuat Grad-CAM untuk kelas: {prediction} di layer: {LAST_CONV_LAYER_NAME}")
                
                try:
                    heatmap = get_gradcam_heatmap(model, img_array, LAST_CONV_LAYER_NAME, pred_index)
                    gradcam_filename = "gradcam_" + filename
                    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                    save_and_display_gradcam(filepath, heatmap, gradcam_path)
                    gradcam_url = gradcam_path
                except Exception as e_gradcam:
                    print(f"Gagal membuat Grad-CAM: {e_gradcam}")
                    # Jangan hentikan aplikasi, cukup biarkan gradcam_url None
                    # Ini berguna jika layer name salah, prediksi tetap jalan.

            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"Terjadi error pada sistem: {str(e)}"

    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence, 
                           img_url=img_url,
                           gradcam_url=gradcam_url,
                           solution=solution_info) # Kirim data solusi ke HTML

if __name__ == '__main__':
    app.run(debug=True, port=5000)