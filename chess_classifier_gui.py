import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
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
        self.root.title("Chess Piece Classifier Pro")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        # Настройка темы
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Загрузка модели
        try:
            self.model = load_model("chess_model_updated.keras")
            self.model_loaded = True
        except Exception as e:
            self.model = None
            self.model_loaded = False
        
        # Классы фигур
        self.class_labels = {
            'Queen-Resized': 'Ферзь 👑',
            'Rook-resize': 'Ладья 🏰',
            'bishop_resized': 'Слон 🐘',
            'knight-resize': 'Конь 🐴',
            'pawn_resized': 'Пешка 🧍‍♂️'
        }
        
        self.log_file = "predictions_log.csv"
        self.create_widgets()
        
    def create_widgets(self):
        # Основной контейнер
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Заголовок
        self.title_label = ctk.CTkLabel(
            self.main_container,
            text="Классификатор шахматных фигур",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=10)
        
        # Статус модели
        status_text = "✅ Модель загружена" if self.model_loaded else "❌ Ошибка загрузки модели"
        status_color = "green" if self.model_loaded else "red"
        self.status_label = ctk.CTkLabel(
            self.main_container,
            text=status_text,
            text_color=status_color,
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=5)
        
        # Контейнер для основного контента
        self.content_frame = ctk.CTkFrame(self.main_container)
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Левая панель (изображение и результаты)
        self.left_panel = ctk.CTkFrame(self.content_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Кнопка загрузки
        self.upload_button = ctk.CTkButton(
            self.left_panel,
            text="Загрузить изображение",
            command=self.upload_image,
            font=ctk.CTkFont(size=14),
            height=40
        )
        self.upload_button.pack(pady=10)
        
        # Фрейм для изображения
        self.image_frame = ctk.CTkFrame(self.left_panel)
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text="Нет загруженного изображения",
            font=ctk.CTkFont(size=14)
        )
        self.image_label.pack(expand=True)
        
        # Фрейм для результатов
        self.result_frame = ctk.CTkFrame(self.left_panel)
        self.result_frame.pack(fill=tk.X, pady=10)
        
        # Правая панель (история)
        self.right_panel = ctk.CTkFrame(self.content_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Заголовок истории
        self.history_label = ctk.CTkLabel(
            self.right_panel,
            text="История классификаций",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.history_label.pack(pady=10)
        
        # Таблица истории
        self.create_history_table()
        
    def create_history_table(self):
        # Фрейм для таблицы с прокруткой
        self.table_frame = ctk.CTkFrame(self.right_panel)
        self.table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создание Treeview с темным стилем
        style = ttk.Style()
        style.configure(
            "Treeview",
            background="#2a2d2e",
            foreground="white",
            fieldbackground="#2a2d2e",
            borderwidth=0
        )
        style.configure(
            "Treeview.Heading",
            background="#2a2d2e",
            foreground="white",
            borderwidth=1
        )
        
        columns = ("Файл", "Класс", "Цвет", "Уверенность")
        self.history_tree = ttk.Treeview(
            self.table_frame,
            columns=columns,
            show="headings",
            style="Treeview"
        )
        
        # Настройка заголовков и столбцов
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=120)
        
        # Добавление полосы прокрутки
        scrollbar = ctk.CTkScrollbar(
            self.table_frame,
            command=self.history_tree.yview
        )
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Загрузка истории
        self.load_history()
    
    def show_result(self, class_name, color, confidence):
        # Очистка предыдущих результатов
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Создание красивых карточек с результатами
        class_card = ctk.CTkFrame(self.result_frame)
        class_card.pack(fill=tk.X, pady=5, padx=10)
        
        ctk.CTkLabel(
            class_card,
            text="Тип фигуры:",
            font=ctk.CTkFont(size=12)
        ).pack()
        
        ctk.CTkLabel(
            class_card,
            text=class_name,
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack()
        
        color_card = ctk.CTkFrame(self.result_frame)
        color_card.pack(fill=tk.X, pady=5, padx=10)
        
        ctk.CTkLabel(
            color_card,
            text="Цвет:",
            font=ctk.CTkFont(size=12)
        ).pack()
        
        ctk.CTkLabel(
            color_card,
            text=color,
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack()
        
        conf_card = ctk.CTkFrame(self.result_frame)
        conf_card.pack(fill=tk.X, pady=5, padx=10)
        
        ctk.CTkLabel(
            conf_card,
            text="Уверенность:",
            font=ctk.CTkFont(size=12)
        ).pack()
        
        # Прогресс-бар уверенности
        confidence_value = float(confidence.strip('%'))
        progress = ctk.CTkProgressBar(conf_card)
        progress.pack(pady=5)
        progress.set(confidence_value / 100)
        
        ctk.CTkLabel(
            conf_card,
            text=f"{confidence_value:.1f}%",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack()
    
    def display_image(self, file_path):
        try:
            # Загрузка и изменение размера изображения
            img = Image.open(file_path)
            img = self.resize_image(img, (300, 300))
            img_tk = ImageTk.PhotoImage(img)
            
            # Обновление метки с изображением
            if hasattr(self, "image_label"):
                self.image_label.destroy()
            
            self.image_label = ctk.CTkLabel(
                self.image_frame,
                image=img_tk,
                text=""
            )
            self.image_label.image = img_tk
            self.image_label.pack(expand=True)
            
        except Exception as e:
            if hasattr(self, "image_label"):
                self.image_label.destroy()
            self.image_label = ctk.CTkLabel(
                self.image_frame,
                text=f"Ошибка отображения: {e}",
                text_color="red"
            )
            self.image_label.pack(expand=True)
    
    def resize_image(self, img, size):
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
            fig_color, _ = self.detect_color(file_path)
            
            # Предсказание класса
            img = image.load_img(file_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            
            prediction = self.model.predict(x)[0]
            idx = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100
            predicted_class = self.class_labels[list(self.class_labels.keys())[idx]]
            
            # Отображение результатов
            self.show_result(predicted_class, fig_color, f"{confidence:.1f}%")
            
            # Сохранение в историю
            self.save_to_history(
                os.path.basename(file_path),
                predicted_class,
                fig_color,
                f"{confidence:.1f}%"
            )
            
        except Exception as e:
            self.show_result("Ошибка", "Ошибка", "0%")
            print(f"Ошибка классификации: {e}")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            self.display_image(file_path)
            if self.model_loaded:
                self.classify_image(file_path)
    
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
    
    def save_to_history(self, file_name, class_name, color, confidence):
        # Добавление в таблицу
        self.history_tree.insert("", 0, values=(file_name, class_name, color, confidence))
        
        # Сохранение в CSV
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
    root = ctk.CTk()
    app = ChessClassifierApp(root)
    root.mainloop() 