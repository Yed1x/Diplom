import os
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def normalize_class_names(input_dir, output_dir):
    """
    Нормализует имена классов и копирует файлы в новую структуру
    """
    # Словарь соответствия старых и новых имен классов
    class_mapping = {
        'bishop_resized': 'bishop',
        'knight-resize': 'knight',
        'pawn_resized': 'pawn',
        'Queen-Resized': 'queen',
        'Rook-resize': 'rook'
    }
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем поддиректории для каждого класса
    for new_class in class_mapping.values():
        os.makedirs(os.path.join(output_dir, new_class), exist_ok=True)
    
    # Копируем и переименовываем файлы
    for old_class, new_class in class_mapping.items():
        old_path = os.path.join(input_dir, old_class)
        new_path = os.path.join(output_dir, new_class)
        
        if not os.path.exists(old_path):
            print(f"Предупреждение: директория {old_path} не найдена")
            continue
            
        for filename in os.listdir(old_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(old_path, filename)
                dst = os.path.join(new_path, filename)
                shutil.copy2(src, dst)
                print(f"Скопирован файл: {src} -> {dst}")

def check_image_quality(directory):
    """
    Проверяет качество изображений в каждой категории
    """
    quality_stats = {}
    
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue
            
        images = []
        corrupted = []
        too_small = []
        
        for filename in os.listdir(class_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            try:
                img_path = os.path.join(class_path, filename)
                img = Image.open(img_path)
                
                # Проверяем размер
                if img.size[0] < 100 or img.size[1] < 100:
                    too_small.append(filename)
                    continue
                    
                # Проверяем, что изображение можно загрузить в массив
                img_array = np.array(img)
                if img_array.size == 0:
                    corrupted.append(filename)
                    continue
                    
                images.append({
                    'filename': filename,
                    'size': img.size,
                    'mode': img.mode,
                    'array_shape': img_array.shape
                })
                
            except Exception as e:
                corrupted.append(filename)
                print(f"Ошибка при обработке {filename}: {str(e)}")
                
        quality_stats[class_name] = {
            'total': len(os.listdir(class_path)),
            'valid': len(images),
            'corrupted': len(corrupted),
            'too_small': len(too_small),
            'samples': images[:5]  # Сохраняем информацию о первых 5 изображениях
        }
    
    return quality_stats

def visualize_samples(directory, num_samples=5):
    """
    Визуализирует примеры изображений из каждой категории
    """
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue
            
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            continue
            
        plt.figure(figsize=(15, 3))
        plt.suptitle(f'Примеры класса: {class_name}')
        
        for i, img_name in enumerate(images[:num_samples]):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path)
                plt.subplot(1, num_samples, i + 1)
                plt.imshow(img)
                plt.title(f'Размер: {img.size}')
                plt.axis('off')
            except Exception as e:
                print(f"Ошибка при отображении {img_name}: {str(e)}")
                
        plt.tight_layout()
        plt.savefig(f'class_samples_{class_name}.png')
        plt.close()

def main():
    # Пути к директориям
    raw_dir = 'data/raw'
    normalized_dir = 'data/normalized'
    
    # Нормализуем имена классов
    print("Нормализация имен классов...")
    normalize_class_names(raw_dir, normalized_dir)
    
    # Проверяем качество изображений
    print("\nПроверка качества изображений...")
    quality_stats = check_image_quality(normalized_dir)
    
    # Выводим статистику
    print("\nСтатистика по классам:")
    for class_name, stats in quality_stats.items():
        print(f"\n{class_name}:")
        print(f"  Всего файлов: {stats['total']}")
        print(f"  Валидных: {stats['valid']}")
        print(f"  Поврежденных: {stats['corrupted']}")
        print(f"  Слишком маленьких: {stats['too_small']}")
        
        if stats['samples']:
            print("\n  Примеры изображений:")
            for sample in stats['samples']:
                print(f"    {sample['filename']}:")
                print(f"      Размер: {sample['size']}")
                print(f"      Режим: {sample['mode']}")
                print(f"      Форма массива: {sample['array_shape']}")
    
    # Визуализируем примеры
    print("\nВизуализация примеров...")
    visualize_samples(normalized_dir)

if __name__ == "__main__":
    main() 