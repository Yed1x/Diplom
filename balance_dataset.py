import os
import shutil
import random

def balance_class(input_dir, output_dir, max_per_class=5000):
    os.makedirs(output_dir, exist_ok=True)
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        selected = images[:max_per_class]
        for img in selected:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_class_path, img)
            shutil.copy2(src, dst)
        print(f"{class_name}: {len(selected)} изображений скопировано.")

def main():
    random.seed(42)
    # Балансируем train
    balance_class('data/augmented_train', 'data/balanced_train', max_per_class=5000)
    # Балансируем val (если нужно)
    balance_class('data/val', 'data/balanced_val', max_per_class=200)

if __name__ == "__main__":
    main() 