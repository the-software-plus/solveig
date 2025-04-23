# model/train.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configuração do Dataset
DATASET_DIR = "data/**"  # Update do dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Pre-processamento de dados
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Definição do modelo
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# base congelada layers
for layer in base_model.layers:
    layer.trainable = False

# Compilar Modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar Modelo
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Salvar Modelo
model.save("model/plant_disease_model.h5")

# Salver Nome das Classes
with open("model/class_names.txt", "w") as f:
    f.write("\n".join(train_generator.class_indices.keys()))