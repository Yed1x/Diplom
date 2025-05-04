import os
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from pathlib import Path

def create_enhanced_augmentation():
    """
    Создает генератор с расширенной аугментацией
    """
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        brightness_range=[0.7, 1.3],
        channel_shift_range=50.0,
        validation_split=0.2
    )

def augment_images(input_dir, output_dir, target_count=200):
    """
    Аугментирует изображения до достижения целевого количества
    """
    datagen = create_enhanced_augmentation()
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        
        # Получаем список изображений
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(images)
        
        print(f"\nОбработка класса {class_name}:")
        print(f"Текущее количество изображений: {current_count}")
        
        if current_count >= target_count:
            print(f"Класс {class_name} уже имеет достаточно изображений")
            continue
            
        # Копируем существующие изображения
        for img_name in images:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(output_class_path, img_name)
            shutil.copy2(src, dst)
        
        # Вычисляем, сколько новых изображений нужно создать
        needed_count = target_count - current_count
        print(f"Нужно создать {needed_count} новых изображений")
        
        # Создаем новые изображения
        for img_name in images:
            if needed_count <= 0:
                break
                
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            img_array = np.array(img)
            img_array = img_array.reshape((1,) + img_array.shape)
            
            # Генерируем новые изображения
            i = 0
            for batch in datagen.flow(img_array, 
                                    batch_size=1,
                                    save_to_dir=output_class_path,
                                    save_prefix=f'{class_name}_aug',
                                    save_format='jpg'):
                i += 1
                needed_count -= 1
                if i >= 5 or needed_count <= 0:  # Создаем до 5 вариантов из каждого изображения
                    break
        
        # Проверяем финальное количество
        final_count = len(os.listdir(output_class_path))
        print(f"Финальное количество изображений: {final_count}")

def verify_dataset(directory):
    """
    Проверяет и выводит статистику по датасету
    """
    stats = {}
    total_images = 0
    
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue
            
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(images)
        stats[class_name] = count
        total_images += count
        
        print(f"\n{class_name}:")
        print(f"Количество изображений: {count}")
        
        # Показываем примеры
        if images:
            plt.figure(figsize=(15, 3))
            plt.suptitle(f'Примеры класса: {class_name}')
            
            for i, img_name in enumerate(images[:5]):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path)
                    plt.subplot(1, 5, i + 1)
                    plt.imshow(img)
                    plt.title(f'Размер: {img.size}')
                    plt.axis('off')
                except Exception as e:
                    print(f"Ошибка при отображении {img_name}: {str(e)}")
                    
            plt.tight_layout()
            plt.savefig(f'enhanced_samples_{class_name}.png')
            plt.close()
    
    print(f"\nОбщее количество изображений: {total_images}")
    print(f"Среднее количество на класс: {total_images / len(stats):.1f}")
    
    # Строим график распределения классов
    plt.figure(figsize=(10, 5))
    plt.bar(stats.keys(), stats.values())
    plt.title('Распределение изображений по классам')
    plt.xlabel('Класс')
    plt.ylabel('Количество изображений')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()

def main():
    # Пути к директориям
    input_dir = 'data/normalized'
    output_dir = 'data/enhanced'
    
    # Аугментируем изображения
    print("Начинаем аугментацию данных...")
    augment_images(input_dir, output_dir, target_count=200)
    
    # Проверяем результат
    print("\nПроверяем результат аугментации...")
    verify_dataset(output_dir)

if __name__ == "__main__":
    main() 