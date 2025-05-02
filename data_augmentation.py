import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def create_augmented_data(input_dir, output_dir, samples_per_image=10):
    """
    Создает аугментированные изображения для каждого класса
    """
    # Создаем генератор с различными преобразованиями
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.7, 1.3]
    )

    # Проходим по каждому классу
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        # Создаем выходную директорию если её нет
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        
        # Проходим по каждому изображению в классе
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_dir, img_name)
            
            # Загружаем изображение
            img = load_img(img_path, target_size=(224, 224))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            
            # Генерируем новые изображения
            i = 0
            for batch in datagen.flow(x, 
                                    batch_size=1,
                                    save_to_dir=output_class_dir,
                                    save_prefix=f'{class_name}_{i}_aug',
                                    save_format='jpg'):
                i += 1
                if i >= samples_per_image:
                    break

def main():
    # Пути к данным
    input_dir = 'data/train'
    output_dir = 'data/augmented_train'
    
    # Создаем аугментированные данные
    create_augmented_data(input_dir, output_dir, samples_per_image=20)
    
    # Выводим статистику
    print("\nСтатистика после аугментации:")
    for class_name in os.listdir(output_dir):
        class_dir = os.path.join(output_dir, class_name)
        n_files = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{class_name}: {n_files} изображений")

if __name__ == "__main__":
    main() 