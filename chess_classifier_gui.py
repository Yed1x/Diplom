import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
from datetime import datetime

class ChessClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Piece Classifier")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Загрузка модели
        try:
            self.model = load_model("chess_model_updated.keras")
            self.status_label = tk.Label(root, text="Модель успешно загружена", fg="green")
        except Exception as e:
            self.model = None
            self.status_label = tk.Label(root, text=f"Ошибка загрузки модели: {e}", fg="red")
        
        self.status_label.pack(pady=10)
        
        # Классы фигур
        self.class_labels = {
            'Queen-Resized': 'Ферзь 👑',
            'Rook-resize': 'Ладья 🏰',
            'bishop_resized': 'Слон 🐘',
            'knight-resize': 'Конь 🐴',
            'pawn_resized': 'Пешка 🧍‍♂️'
        }
        
        # Лог-файл
        self.log_file = "predictions_log.csv"
        
        # Создание интерфейса
        self.create_widgets()
        
    def create_widgets(self):
        # Фрейм для кнопок
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Кнопка загрузки изображения
        self.upload_button = tk.Button(button_frame, text="Загрузить изображение", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=10)
        
        # Фрейм для изображения
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10)
        
        # Заглушка для изображения
        self.image_label = tk.Label(self.image_frame, text="Нет загруженного изображения")
        self.image_label.pack()
        
        # Фрейм для результатов
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Таблица для отображения истории
        self.create_history_table()
    
    def create_history_table(self):
        columns = ("Файл", "Класс", "Цвет", "Уверенность")
        self.history_tree = ttk.Treeview(self.results_frame, columns=columns, show="headings")
        
        # Настройка заголовков
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100)
        
        # Добавление полосы прокрутки
        scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar.set)
        
        # Размещение элементов
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Загрузка истории из файла, если он существует
        self.load_history()
    
    def load_history(self):
        if os.path.exists(self.log_file):
            try:
                df_log = pd.read_csv(self.log_file, encoding='utf-8', errors='replace')
                
                # Очистка таблицы
                for item in self.history_tree.get_children():
                    self.history_tree.delete(item)
                
                # Заполнение таблицы
                for _, row in df_log.iterrows():
                    self.history_tree.insert("", tk.END, values=(
                        row.get("Файл", ""),
                        row.get("Класс", ""),
                        row.get("Цвет", ""),
                        row.get("Уверенность", "")
                    ))
            except Exception as e:
                print(f"Ошибка загрузки истории: {e}")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        # Отображение изображения
        self.display_image(file_path)
        
        # Классификация изображения
        if self.model:
            self.classify_image(file_path)
    
    def display_image(self, file_path):
        try:
            # Загрузка и изменение размера для отображения
            img = Image.open(file_path)
            img = self.resize_image(img, (300, 300))
            img_tk = ImageTk.PhotoImage(img)
            
            # Обновление метки с изображением
            if hasattr(self, "image_label"):
                self.image_label.destroy()
            
            self.image_label = tk.Label(self.image_frame, image=img_tk)
            self.image_label.image = img_tk  # Предотвращение удаления сборщиком мусора
            self.image_label.pack()
            
        except Exception as e:
            if hasattr(self, "image_label"):
                self.image_label.destroy()
            self.image_label = tk.Label(self.image_frame, text=f"Ошибка отображения: {e}")
            self.image_label.pack()
    
    def resize_image(self, img, size):
        # Сохранение пропорций при изменении размера
        width, height = img.size
        ratio = min(size[0]/width, size[1]/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    def detect_color(self, file_path):
        try:
            img = Image.open(file_path).convert("L")
            arr = np.array(img)
            h, w = arr.shape
            cx, cy = w // 2, h // 2
            s = min(h, w) // 2
            crop = arr[cy - s//2:cy + s//2, cx - s//2:cx + s//2]
            crop_valid = crop[crop < 240]
            
            if len(crop_valid) == 0:
                return "Не удалось определить цвет ❔", 0
            
            mean = np.mean(crop_valid)
            color = "Чёрная ♟️" if mean < 127 else "Белая ♙"
            return color, mean
        except Exception as e:
            return f"Ошибка определения цвета: {e}", 0
    
    def classify_image(self, file_path):
        try:
            # Определение цвета
            fig_color, brightness = self.detect_color(file_path)
            
            # Предсказание класса
            img = image.load_img(file_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            
            prediction = self.model.predict(x)[0]
            idx = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100
            predicted_class = self.class_labels[list(self.class_labels.keys())[idx]]
            
            # Создание фрейма для результатов, если его еще нет
            if hasattr(self, "result_label"):
                self.result_label.destroy()
            
            # Отображение результата
            result_text = f"Класс: {predicted_class}\nЦвет: {fig_color}\nУверенность: {confidence:.2f}%"
            self.result_label = tk.Label(self.image_frame, text=result_text, font=("Arial", 12))
            self.result_label.pack(pady=10)
            
            # Сохранение в историю
            self.save_to_history(os.path.basename(file_path), predicted_class, fig_color, f"{confidence:.2f}%")
            
        except Exception as e:
            if hasattr(self, "result_label"):
                self.result_label.destroy()
            self.result_label = tk.Label(self.image_frame, text=f"Ошибка классификации: {e}", fg="red")
            self.result_label.pack(pady=10)
    
    def save_to_history(self, file_name, class_name, color, confidence):
        # Добавление в таблицу
        self.history_tree.insert("", 0, values=(file_name, class_name, color, confidence))
        
        # Сохранение в CSV
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = pd.DataFrame([{
            "Файл": file_name,
            "Класс": class_name,
            "Цвет": color,
            "Уверенность": confidence
        }])
        
        if os.path.exists(self.log_file):
            try:
                current = pd.read_csv(self.log_file, encoding='utf-8')
                if set(current.columns) == set(new_entry.columns):
                    new_entry.to_csv(self.log_file, mode='a', index=False, header=False, encoding='utf-8-sig')
                else:
                    new_entry.to_csv(self.log_file, index=False, encoding='utf-8-sig')
            except:
                new_entry.to_csv(self.log_file, index=False, encoding='utf-8-sig')
        else:
            new_entry.to_csv(self.log_file, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessClassifierApp(root)
    root.mainloop() 