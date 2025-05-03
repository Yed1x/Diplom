import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

def inspect_folder(folder, classes, n_show=5):
    print(f'Папка: {folder}')
    for class_name in classes:
        class_path = os.path.join(folder, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f'  {class_name}: {len(images)} файлов')
        # Показываем примеры
        if len(images) > 0:
            sample_imgs = random.sample(images, min(n_show, len(images)))
            fig, axes = plt.subplots(1, len(sample_imgs), figsize=(12, 3))
            fig.suptitle(f'{class_name} ({folder})')
            for ax, img_name in zip(axes, sample_imgs):
                img_path = os.path.join(class_path, img_name)
                img = load_img(img_path)
                ax.imshow(img)
                ax.axis('off')
            plt.show()
        else:
            print(f'    Нет изображений для класса {class_name}')

def main():
    random.seed(42)
    classes = ['bishop', 'knight', 'pawn', 'queen', 'rook']
    inspect_folder('data/balanced_train', classes)
    inspect_folder('data/balanced_val', classes)

if __name__ == '__main__':
    main() 