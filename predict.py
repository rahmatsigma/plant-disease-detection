import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# --- KONFIGURASI ---
MODEL_PATH = 'models/plant_disease_model.h5'
CLASS_NAMES_PATH = 'models/class_names.txt'

def load_class_names():
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def predict_image(img_path):
    # 1. Load Model
    print("Memuat model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    # 2. Preprocessing Gambar (Harus sama persis dengan saat training)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Ubah jadi (1, 224, 224, 3)
    img_array /= 255.0 # Normalisasi

    # 3. Prediksi
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)

    print("------------------------------------------------")
    print(f"Hasil Prediksi: {predicted_class}")
    print(f"Tingkat Keyakinan (Confidence): {confidence:.2f}%")
    print("------------------------------------------------")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cara pakai: python predict.py <path_gambar>")
    else:
        img_path = sys.argv[1]
        predict_image(img_path)