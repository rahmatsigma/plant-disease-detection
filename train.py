import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# --- KONFIGURASI ---
DATASET_DIR = 'dataset/train' 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = 'models/plant_disease_model.h5'

def train_model():
    # 1. Data Augmentation (Agar model lebih pintar mengenali variasi gambar)
    train_datagen = ImageDataGenerator(
        rescale=1./255,             
        rotation_range=20,         
        width_shift_range=0.2,     
        height_shift_range=0.2,     
        horizontal_flip=True,       
        validation_split=0.2        
    )

    # 2. Load Data Training
    print("Memuat Data Training...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # 3. Load Data Validasi
    print("Memuat Data Validasi...")
    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    #label kelas 
    class_names = list(train_generator.class_indices.keys())
    print(f"Kelas yang ditemukan: {class_names}")
    
    #nama kelas 
    with open('models/class_names.txt', 'w') as f:
        f.write('\n'.join(class_names))

    # 4. Membangun Model 
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Bekukan base_model 
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) 
    predictions = Dense(len(class_names), activation='softmax')(x) 

    model = Model(inputs=base_model.input, outputs=predictions)

    # 5. Compile Model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 6. Training
    print("Mulai Training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # 7. Simpan Model
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save(MODEL_SAVE_PATH)
    print(f"Model berhasil disimpan di {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()