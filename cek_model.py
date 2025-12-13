import tensorflow as tf

MODEL_PATH = 'models/plant_disease_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Kita mencari layer Conv2D terakhir sebelum GlobalAveragePooling2D
# Biasanya di MobileNetV2, ini adalah layer bernama 'out_relu' yang merupakan bagian dari base_model
print("=== DAFTAR LAYER ===")
for layer in model.layers:
    print(layer.name)

# Jika Anda melihat 'mobilenetv2_1.00_224' (atau sejenisnya), kita perlu melihat ke dalamnya
base_model = model.get_layer(index=0) # Asumsi base model ada di layer pertama
print("\n=== LAYER AKHIR BASE MODEL ===")
# Kita ambil 5 layer terakhir dari base model untuk dicek
for layer in base_model.layers[-5:]:
    print(layer.name)