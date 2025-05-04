import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_augmentation_generator():
    """
    Создает генератор для аугментации изображений с различными трансформациями
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
        brightness_range=[0.8, 1.2],
        channel_shift_range=50.0
    )

def augment_class(input_dir, output_dir, class_name, target_count):
    """
    Аугментирует изображения класса до целевого количества
    """
    class_path = os.path.join(input_dir, class_name)
    output_path = os.path.join(output_dir, class_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Получаем список изображений
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(images)
    
    if current_count >= target_count:
        print(f"Класс {class_name} уже имеет достаточно изображений ({current_count})")
        return
    
    # Копируем оригинальные изображения
    for img_name in images:
        src_path = os.path.join(class_path, img_name)
        dst_path = os.path.join(output_path, img_name)
        Image.open(src_path).save(dst_path)
    
    # Создаем генератор аугментации
    datagen = create_augmentation_generator()
    
    # Аугментируем изображения
    needed_count = target_count - current_count
    augmentations_per_image = needed_count // current_count + 1
    
    print(f"\nАугментация класса {class_name}:")
    print(f"Текущее количество: {current_count}")
    print(f"Целевое количество: {target_count}")
    print(f"Нужно создать: {needed_count}")
    print(f"Аугментаций на изображение: {augmentations_per_image}")
    
    for img_name in tqdm(images, desc=f"Аугментация {class_name}"):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path)
        img_array = np.array(img)
        img_array = img_array.reshape((1,) + img_array.shape)
        
        # Генерируем аугментированные изображения
        aug_iter = datagen.flow(
            img_array,
            batch_size=1,
            save_to_dir=output_path,
            save_prefix=f'aug_{os.path.splitext(img_name)[0]}',
            save_format='png'
        )
        
        for _ in range(augmentations_per_image):
            next(aug_iter)
    
    # Проверяем результат
    final_count = len([f for f in os.listdir(output_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Итоговое количество изображений: {final_count}")

def visualize_augmentations(input_dir, output_dir, class_name):
    """
    Визуализирует примеры аугментации для класса
    """
    class_path = os.path.join(input_dir, class_name)
    output_path = os.path.join(output_dir, class_name)
    
    # Получаем оригинальное изображение
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        return
    
    original_img = Image.open(os.path.join(class_path, images[0]))
    
    # Получаем аугментированные изображения
    aug_images = [f for f in os.listdir(output_path) if f.startswith('aug_')][:4]
    
    # Создаем визуализацию
    plt.figure(figsize=(15, 3))
    plt.suptitle(f'Примеры аугментации для класса: {class_name}')
    
    # Показываем оригинал
    plt.subplot(1, 5, 1)
    plt.imshow(original_img)
    plt.title('Оригинал')
    plt.axis('off')
    
    # Показываем аугментации
    for i, img_name in enumerate(aug_images):
        img = Image.open(os.path.join(output_path, img_name))
        plt.subplot(1, 5, i + 2)
        plt.imshow(img)
        plt.title(f'Аугментация {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'augmentation_samples_{class_name}.png', dpi=100)
    plt.close()

def main():
    input_dir = 'data/merged'
    output_dir = 'data/augmented'
    target_count = 1000  # Целевое количество изображений для каждого класса
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список классов
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Аугментируем каждый класс
    for class_name in classes:
        augment_class(input_dir, output_dir, class_name, target_count)
        visualize_augmentations(input_dir, output_dir, class_name)
    
    print("\nАугментация завершена!")
    print(f"Результаты сохранены в директории: {output_dir}")

if __name__ == "__main__":
    main() 