import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Параметры обучения
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5

def create_model():
    """
    Создает модель на основе MobileNetV2 с предварительно обученными весами
    """
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Замораживаем базовую модель
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_data_generators():
    """
    Создает генераторы данных для обучения и валидации
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'data/augmented',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        'data/merged',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator

def create_callbacks():
    """
    Создает callbacks для обучения
    """
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]

def plot_training_history(history):
    """
    Строит графики точности и потерь
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # График точности
    ax1.plot(history.history['accuracy'], label='Обучающая выборка')
    ax1.plot(history.history['val_accuracy'], label='Валидационная выборка')
    ax1.set_title('Точность модели')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    
    # График потерь
    ax2.plot(history.history['loss'], label='Обучающая выборка')
    ax2.plot(history.history['val_loss'], label='Валидационная выборка')
    ax2.set_title('Потери модели')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Создаем модель
    model = create_model()
    
    # Компилируем модель
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Создаем генераторы данных
    train_generator, val_generator = create_data_generators()
    
    # Создаем callbacks
    callbacks = create_callbacks()
    
    # Обучаем модель
    print("\nНачинаем обучение модели...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Строим графики обучения
    plot_training_history(history)
    
    # Разблокируем fine-tuning
    print("\nНачинаем fine-tuning...")
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Замораживаем первые слои
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Перекомпилируем модель с меньшей скоростью обучения
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Продолжаем обучение
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # Сохраняем финальную модель
    model.save('final_model.h5')
    print("\nОбучение завершено!")
    print("Модели сохранены как 'best_model.h5' и 'final_model.h5'")
    print("График обучения сохранен как 'training_history.png'")

if __name__ == "__main__":
    main() 