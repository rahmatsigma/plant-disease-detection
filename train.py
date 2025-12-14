import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# --- KONFIGURASI (HYPERPARAMETERS) ---
IMG_SIZE = (224, 224)  # Resolusi standar EfficientNet
BATCH_SIZE = 32
DATA_DIR = 'dataset/train' # Pastikan path ini sesuai dengan folder dataset Anda
LEARNING_RATE = 0.001
EPOCHS_HEAD = 10       # Epoch untuk pelatihan tahap 1 (Head only)
EPOCHS_FINE = 10       # Epoch untuk pelatihan tahap 2 (Fine tuning)

def main():
    # 1. SETUP DATA GENERATOR (ADVANCED AUGMENTATION)
    # Teknik ini membuat variasi gambar buatan agar dataset terasa lebih banyak
    train_datagen = ImageDataGenerator(
        rescale=1./255,             # Normalisasi pixel
        rotation_range=40,          # Putar gambar hingga 40 derajat
        width_shift_range=0.2,      # Geser horizontal
        height_shift_range=0.2,     # Geser vertikal
        shear_range=0.2,            # Efek miring (shear)
        zoom_range=0.2,             # Zoom in/out
        horizontal_flip=True,       # Balik horizontal
        brightness_range=[0.8, 1.2],# Variasi kecerahan
        fill_mode='nearest',
        validation_split=0.2        # 20% data untuk validasi otomatis
    )

    print("Load Data Training...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    print("Load Data Validation...")
    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = train_generator.num_classes
    class_names = list(train_generator.class_indices.keys())
    print(f"Kelas terdeteksi: {class_names}")

    # Simpan nama kelas untuk dipakai di app.py nanti
    if not os.path.exists('models'):
        os.makedirs('models')
    with open('models/class_names.txt', 'w') as f:
        f.write('\n'.join(class_names))

    # 2. MEMBANGUN MODEL (TRANSFER LEARNING)
    # Menggunakan EfficientNetB0 yang sudah pre-trained dengan ImageNet
    print("\nMembangun Model EfficientNetB0...")
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Bekukan base model (Freeze) agar bobot cerdasnya tidak rusak saat awal training
    base_model.trainable = False 

    # Membuat 'Custom Head' (Bagian otak baru khusus penyakit tanaman)
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Lebih bagus daripada Flatten
    x = BatchNormalization()(x)     # Menstabilkan training
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)             # Mencegah Overfitting (matikan 30% neuron secara acak)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile Model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks (Fitur Otomatis)
    callbacks = [
        # Simpan model terbaik saja berdasarkan val_accuracy
        ModelCheckpoint('models/best_model_v2.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
        # Stop jika akurasi tidak naik selama 5 epoch (hemat waktu)
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        # Kurangi learning rate jika stuck
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    # 3. TRAINING TAHAP 1 (Feature Extraction)
    print(f"\n--- TAHAP 1: Training Head ({EPOCHS_HEAD} Epochs) ---")
    history_head = model.fit(
        train_generator,
        epochs=EPOCHS_HEAD,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # 4. TRAINING TAHAP 2 (Fine-Tuning)
    # Ini langkah PRO: Kita cairkan (unfreeze) model dasar dan latih ulang pelan-pelan
    print("\n--- TAHAP 2: Fine-Tuning (Meningkatkan Akurasi) ---")
    base_model.trainable = True
    
    # Kita hanya latih ulang beberapa layer teratas dari base model (opsional, di sini kita unfreeze semua tapi LR kecil)
    # Penting: Gunakan Learning Rate yang SANGAT KECIL agar bobot lama tidak hancur
    model.compile(optimizer=Adam(learning_rate=1e-5), # 1e-5 sangat kecil
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    total_epochs = EPOCHS_HEAD + EPOCHS_FINE
    
    history_fine = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=history_head.epoch[-1], # Lanjut dari epoch terakhir
        validation_data=val_generator,
        callbacks=callbacks
    )

    print("\nTraining Selesai! Model disimpan di folder models/")
    
    # 5. VISUALISASI HASIL
    acc = history_head.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history_head.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history_head.history['loss'] + history_fine.history['loss']
    val_loss = history_head.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.5, 1])
    plt.plot([EPOCHS_HEAD-1, EPOCHS_HEAD-1], plt.ylim(), label='Mulai Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([EPOCHS_HEAD-1, EPOCHS_HEAD-1], plt.ylim(), label='Mulai Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('training_result.png')
    print("Grafik hasil training disimpan sebagai training_result.png")

if __name__ == '__main__':
    main()