import os
import shutil
import requests
import zipfile
import kaggle
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import multiprocessing

# Устанавливаем количество процессов равным количеству ядер
NUM_PROCESSES = multiprocessing.cpu_count()

def download_kaggle_dataset():
    """
    Загружает датасет с шахматными фигурами с Kaggle
    """
    kaggle_dir = os.path.abspath('data/kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Проверяем, есть ли уже распакованные данные
    if os.path.exists(os.path.join(kaggle_dir, 'dataset')):
        print(f"\nНайдены уже распакованные данные в {kaggle_dir}")
        return
    
    # Загружаем датасет с шахматными фигурами
    print("Загрузка датасета с Kaggle...")
    kaggle.api.dataset_download_files(
        'koryakinp/chess-positions',
        path=kaggle_dir,
        unzip=False
    )
    
    # Распаковываем архив
    zip_path = os.path.join(kaggle_dir, 'chess-positions.zip')
    if os.path.exists(zip_path):
        print(f"\nРаспаковка архива {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for file in tqdm(file_list, desc="Распаковка"):
                zip_ref.extract(file, kaggle_dir)
        
        os.remove(zip_path)

def download_uci_dataset():
    """
    Загружает датасет с UCI
    """
    # URL для загрузки датасета с UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/chess-pieces/chess-pieces.zip"
    response = requests.get(url)
    
    with open('data/uci.zip', 'wb') as f:
        f.write(response.content)
    
    with zipfile.ZipFile('data/uci.zip', 'r') as zip_ref:
        zip_ref.extractall('data/uci')

def normalize_image(img_path, target_size=(224, 224)):
    """
    Нормализует изображение: изменяет размер, конвертирует в RGB
    """
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(target_size, Image.LANCZOS)
        return img
    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {str(e)}")
        return None

def process_image_batch(args):
    """
    Обрабатывает пакет изображений параллельно
    """
    img_paths, output_dir, class_name, datagen, target_count = args
    processed_images = []
    
    for img_path in img_paths:
        normalized_img = normalize_image(img_path)
        if normalized_img is not None:
            processed_images.append(normalized_img)
            
            # Сохраняем оригинальное изображение
            output_path = os.path.join(output_dir, f'original_{len(processed_images):04d}.jpg')
            normalized_img.save(output_path, quality=95, optimize=True)
            
            # Аугментируем изображение
            if len(processed_images) < target_count:
                img_array = np.array(normalized_img)
                img_array = img_array.reshape((1,) + img_array.shape)
                
                i = 0
                for batch in datagen.flow(img_array, 
                                        batch_size=1,
                                        save_to_dir=output_dir,
                                        save_prefix=f'{class_name}_aug',
                                        save_format='jpg'):
                    i += 1
                    if i >= 5 or len(processed_images) >= target_count:
                        break
    
    return processed_images

def merge_datasets(input_dirs, output_dir, target_count=1000):
    """
    Объединяет данные из разных источников
    """
    os.makedirs(output_dir, exist_ok=True)
    
    class_images = {
        'bishop': [],
        'knight': [],
        'pawn': [],
        'queen': [],
        'rook': []
    }
    
    # Создаем аугментатор с оптимизированными параметрами
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # Создаем пул процессов
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        for input_dir in input_dirs:
            for class_name in os.listdir(input_dir):
                class_path = os.path.join(input_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                    
                normalized_class = class_name.lower().replace('-', '').replace('_', '')
                if normalized_class not in class_images:
                    continue
                
                class_dir = os.path.join(output_dir, normalized_class)
                os.makedirs(class_dir, exist_ok=True)
                
                img_paths = [os.path.join(class_path, f) for f in os.listdir(class_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                print(f"\nОбработка класса {normalized_class}:")
                print(f"Найдено изображений: {len(img_paths)}")
                
                # Разбиваем на пакеты для параллельной обработки
                batch_size = max(1, len(img_paths) // NUM_PROCESSES)
                batches = [img_paths[i:i + batch_size] for i in range(0, len(img_paths), batch_size)]
                
                # Обрабатываем пакеты параллельно
                args = [(batch, class_dir, normalized_class, datagen, target_count) 
                        for batch in batches]
                results = list(tqdm(pool.imap(process_image_batch, args), 
                                  total=len(batches),
                                  desc=f"Обработка {normalized_class}"))
                
                # Объединяем результаты
                for batch_result in results:
                    class_images[normalized_class].extend(batch_result)
                
                final_count = len(os.listdir(class_dir))
                print(f"Финальное количество изображений: {final_count}")

def verify_merged_dataset(directory):
    """
    Проверяет и визуализирует объединенный датасет
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
            plt.savefig(f'merged_samples_{class_name}.png', dpi=100)
            plt.close()
    
    print(f"\nОбщее количество изображений: {total_images}")
    print(f"Среднее количество на класс: {total_images / len(stats):.1f}")
    
    plt.figure(figsize=(10, 5))
    plt.bar(stats.keys(), stats.values())
    plt.title('Распределение изображений по классам')
    plt.xlabel('Класс')
    plt.ylabel('Количество изображений')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('merged_distribution.png', dpi=100)
    plt.close()

def main():
    # Включаем оптимизации TensorFlow
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': True,
        'constant_folding': True,
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
        'dependency_optimization': True,
        'loop_optimization': True,
        'function_optimization': True,
        'debug_stripper': True,
        'disable_model_pruning': True,
        'scoped_allocator_optimization': True,
        'pin_to_host_optimization': True,
        'implementation_selector': True,
        'auto_mixed_precision': True
    })
    
    # Создаем директорию для данных
    os.makedirs('data', exist_ok=True)
    
    # Загружаем датасет
    print("Загрузка датасета...")
    try:
        download_kaggle_dataset()
    except Exception as e:
        print(f"Ошибка при загрузке датасета с Kaggle: {str(e)}")
    
    # Объединяем датасеты
    print("\nОбъединение датасетов...")
    input_dirs = [
        'data/kaggle/dataset',
        'data/normalized'
    ]
    
    merge_datasets(input_dirs, 'data/merged', target_count=1000)
    
    # Проверяем результат
    print("\nПроверка объединенного датасета...")
    verify_merged_dataset('data/merged')

if __name__ == "__main__":
    main() 