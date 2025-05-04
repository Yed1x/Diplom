import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def check_dataset(directory):
    """
    Проверяет датасет и выводит статистику
    """
    stats = {}
    total_images = 0
    
    print("\nПроверка датасета:")
    print("-" * 50)
    
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
        
        # Проверяем размеры и формат изображений
        if images:
            sizes = []
            for img_name in images[:100]:  # Проверяем первые 100 изображений
                try:
                    img_path = os.path.join(class_path, img_name)
                    img = Image.open(img_path)
                    sizes.append(img.size)
                except Exception as e:
                    print(f"Ошибка при проверке {img_name}: {str(e)}")
            
            if sizes:
                sizes = np.array(sizes)
                print(f"Средний размер: {np.mean(sizes, axis=0).astype(int)}")
                print(f"Минимальный размер: {np.min(sizes, axis=0)}")
                print(f"Максимальный размер: {np.max(sizes, axis=0)}")
        
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
            plt.savefig(f'dataset_samples_{class_name}.png', dpi=100)
            plt.close()
    
    print("\nОбщая статистика:")
    print("-" * 50)
    print(f"Общее количество изображений: {total_images}")
    print(f"Среднее количество на класс: {total_images / len(stats):.1f}")
    
    # Строим график распределения классов
    plt.figure(figsize=(10, 5))
    plt.bar(stats.keys(), stats.values())
    plt.title('Распределение изображений по классам')
    plt.xlabel('Класс')
    plt.ylabel('Количество изображений')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=100)
    plt.close()

if __name__ == "__main__":
    check_dataset('data/augmented') 