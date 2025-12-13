import numpy as np
import tensorflow as tf
import cv2

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Menghitung heatmap Grad-CAM untuk input gambar tertentu.
    """
    # 1. Kita perlu mengakses output dari layer konvolusi terakhir
    #    dan output prediksi akhir model.
    
    # Jika model adalah transfer learning (Sequential/Functional dengan base model di dalamnya)
    # Kita perlu mengakses base modelnya dulu untuk mendapatkan last_conv_layer
    try:
        # Mencoba mengakses base model jika ada (biasanya layer pertama)
        base_model = model.get_layer(index=0) 
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        last_conv_layer_output = last_conv_layer.output
    except:
         # Fallback jika struktur model berbeda, coba akses langsung
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_output = last_conv_layer.output

    # Buat model "pencari gradient" yang outputnya ada dua:
    # (1) Aktivasi layer konvolusi terakhir, (2) Prediksi akhir model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer_output, model.output]
    )

    # 2. Rekam operasi gradient menggunakan GradientTape
    with tf.GradientTape() as tape:
        # Forward pass gambar melalui model pencari gradient
        last_conv_layer_output, preds = grad_model(img_array)
        
        if pred_index is None:
            # Jika index tidak ditentukan, gunakan kelas dengan probabilitas tertinggi
            pred_index = tf.argmax(preds[0])
        
        # Ambil nilai loss (skor) untuk kelas yang diprediksi
        class_channel = preds[:, pred_index]

    # 3. Hitung Gradient
    # Ini adalah inti Grad-CAM: Bagaimana 'class_channel' berubah terhadap 'last_conv_layer_output'
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Global Average Pooling pada gradients
    # Ini memberikan "bobot pentingnya" setiap filter di layer konvolusi terakhir
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Kalikan output layer konvolusi dengan bobot gradient
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU (Hanya tertarik pada pengaruh positif)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Menggabungkan heatmap dengan gambar asli dan menyimpannya.
    """
    # Load gambar asli dengan OpenCV
    img = cv2.imread(img_path)
    
    # Resize heatmap agar ukurannya sama dengan gambar asli
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Ubah heatmap menjadi RGB menggunakan colormap (misal: JET)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Gabungkan gambar asli dengan heatmap (superimpose)
    # alpha mengatur transparansi heatmap
    superimposed_img = heatmap * alpha + img
    
    # Simpan hasilnya
    cv2.imwrite(cam_path, superimposed_img)