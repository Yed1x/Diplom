import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from idlelib.tooltip import Hovertip

class ChessClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Piece Classifier Pro")
        self.root.geometry("1200x800")
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
        
        # Инициализация статистики
        self.stats = {
            "total_classifications": 0,
            "by_class": {},
            "by_color": {"Белая ♙": 0, "Чёрная ♟️": 0}
        }
        
        self.load_stats()
        self.setup_menu()
        self.create_widgets()
        self.add_tooltips()
        self.batch_process()
        self.add_settings()
        self.compare_images()
        
    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Меню файла
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Открыть изображение", command=self.upload_image)
        file_menu.add_command(label="Экспорт истории", command=self.export_history)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        # Меню статистики
        stats_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Статистика", menu=stats_menu)
        stats_menu.add_command(label="Показать статистику", command=self.show_statistics)
        stats_menu.add_command(label="Сбросить статистику", command=self.reset_statistics)
    
    def create_widgets(self):
        # Основной контейнер
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Панель инструментов
        self.toolbar = ctk.CTkFrame(self.main_container)
        self.toolbar.pack(fill=tk.X, pady=(0, 10))
        
        # Кнопка загрузки с иконкой
        self.upload_button = ctk.CTkButton(
            self.toolbar,
            text="Загрузить изображение",
            command=self.upload_image,
            font=ctk.CTkFont(size=14),
            height=40,
            width=200
        )
        self.upload_button.pack(side=tk.LEFT, padx=5)
        
        # Кнопка статистики
        self.stats_button = ctk.CTkButton(
            self.toolbar,
            text="Показать статистику",
            command=self.show_statistics,
            font=ctk.CTkFont(size=14),
            height=40,
            width=200
        )
        self.stats_button.pack(side=tk.LEFT, padx=5)
        
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
            fig_color, _ = self.detect_color(file_path)
            
            img = image.load_img(file_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            
            prediction = self.model.predict(x)[0]
            idx = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100
            predicted_class = self.class_labels[list(self.class_labels.keys())[idx]]
            
            self.show_result(predicted_class, fig_color, f"{confidence:.1f}%")
            self.save_to_history(
                os.path.basename(file_path),
                predicted_class,
                fig_color,
                f"{confidence:.1f}%"
            )
            
            # Обновляем статистику
            self.update_stats(predicted_class, fig_color)
            
        except Exception as e:
            self.show_result("Ошибка", "Ошибка", "0%")
            print(f"Ошибка классификации: {e}")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png"),
                ("Все файлы", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.process_image(file_path)
            except Exception as e:
                messagebox.showerror(
                    "Ошибка",
                    f"Не удалось обработать изображение:\n{str(e)}"
                )
    
    def load_history(self):
        if os.path.exists(self.log_file):
            try:
                # Исправляем чтение CSV файла
                df_log = pd.read_csv(self.log_file, encoding='utf-8')
                
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
    
    def process_image(self, file_path):
        self.display_image(file_path)
        if self.model_loaded:
            self.classify_image(file_path)
    
    def export_history(self):
        if not os.path.exists(self.log_file):
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[
                ("Excel файл", "*.xlsx"),
                ("CSV файл", "*.csv"),
                ("JSON файл", "*.json")
            ]
        )
        
        if not file_path:
            return
            
        df = pd.read_csv(self.log_file, encoding='utf-8')
        
        if file_path.endswith('.xlsx'):
            df.to_excel(file_path, index=False)
        elif file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        elif file_path.endswith('.json'):
            df.to_json(file_path, orient='records', force_ascii=False)
            
        messagebox.showinfo("Успех", "Данные успешно экспортированы")
    
    def show_statistics(self):
        stats_window = ctk.CTkToplevel(self.root)
        stats_window.title("Статистика классификаций")
        stats_window.geometry("800x600")
        
        # Создаем вкладки
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Вкладка общей статистики
        general_frame = ctk.CTkFrame(notebook)
        notebook.add(general_frame, text="Общая статистика")
        
        total_label = ctk.CTkLabel(
            general_frame,
            text=f"Всего классификаций: {self.stats['total_classifications']}",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        total_label.pack(pady=20)
        
        # График распределения по классам
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        classes = list(self.stats['by_class'].keys())
        values = list(self.stats['by_class'].values())
        ax1.bar(classes, values)
        ax1.set_title("Распределение по классам фигур")
        ax1.tick_params(axis='x', rotation=45)
        
        canvas1 = FigureCanvasTkAgg(fig1, general_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # График распределения по цветам
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors = list(self.stats['by_color'].keys())
        color_values = list(self.stats['by_color'].values())
        ax2.pie(color_values, labels=colors, autopct='%1.1f%%')
        ax2.set_title("Распределение по цветам")
        
        canvas2 = FigureCanvasTkAgg(fig2, general_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def reset_statistics(self):
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите сбросить статистику?"):
            self.stats = {
                "total_classifications": 0,
                "by_class": {},
                "by_color": {"Белая ♙": 0, "Чёрная ♟️": 0}
            }
            self.save_stats()
            messagebox.showinfo("Успех", "Статистика сброшена")
    
    def load_stats(self):
        try:
            if os.path.exists('stats.json'):
                with open('stats.json', 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки статистики: {e}")
    
    def save_stats(self):
        try:
            with open('stats.json', 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения статистики: {e}")
    
    def update_stats(self, class_name, color):
        self.stats['total_classifications'] += 1
        
        if class_name not in self.stats['by_class']:
            self.stats['by_class'][class_name] = 0
        self.stats['by_class'][class_name] += 1
        
        if color in self.stats['by_color']:
            self.stats['by_color'][color] += 1
            
        self.save_stats()

    def add_tooltips(self):
        Hovertip(self.upload_button, "Загрузить изображение для классификации")
        Hovertip(self.stats_button, "Просмотр статистики классификаций")

    def batch_process(self):
        # Добавляем кнопку пакетной обработки в toolbar
        self.batch_button = ctk.CTkButton(
            self.toolbar,
            text="Пакетная обработка",
            command=self.open_batch_window,
            font=ctk.CTkFont(size=14),
            height=40,
            width=200
        )
        self.batch_button.pack(side=tk.LEFT, padx=5)

    def open_batch_window(self):
        folder = filedialog.askdirectory(title="Выберите папку с изображениями")
        if folder:
            progress_window = ctk.CTkToplevel(self.root)
            progress_window.title("Обработка изображений")
            progress_window.geometry("400x200")
            
            progress_label = ctk.CTkLabel(
                progress_window,
                text="Обработка изображений...",
                font=ctk.CTkFont(size=14)
            )
            progress_label.pack(pady=20)
            
            progress_bar = ctk.CTkProgressBar(progress_window)
            progress_bar.pack(pady=10)
            progress_bar.set(0)

    def add_settings(self):
        # Добавляем кнопку настроек в toolbar
        self.settings_button = ctk.CTkButton(
            self.toolbar,
            text="Настройки",
            command=self.open_settings,
            font=ctk.CTkFont(size=14),
            height=40,
            width=200
        )
        self.settings_button.pack(side=tk.LEFT, padx=5)

    def open_settings(self):
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("Настройки")
        settings_window.geometry("500x400")
        
        # Настройки темы
        theme_frame = ctk.CTkFrame(settings_window)
        theme_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(theme_frame, text="Тема оформления:").pack(side=tk.LEFT, padx=5)
        theme_var = tk.StringVar(value="dark")
        theme_menu = ctk.CTkOptionMenu(
            theme_frame,
            values=["dark", "light"],
            variable=theme_var,
            command=lambda x: ctk.set_appearance_mode(x)
        )
        theme_menu.pack(side=tk.LEFT, padx=5)

    def compare_images(self):
        # Добавляем кнопку сравнения в toolbar
        self.compare_button = ctk.CTkButton(
            self.toolbar,
            text="Сравнить изображения",
            command=self.open_compare_window,
            font=ctk.CTkFont(size=14),
            height=40,
            width=200
        )
        self.compare_button.pack(side=tk.LEFT, padx=5)

    def open_compare_window(self):
        compare_window = ctk.CTkToplevel(self.root)
        compare_window.title("Сравнение изображений")
        compare_window.geometry("1000x600")
        
        # Создаем фрейм для изображений
        images_frame = ctk.CTkFrame(compare_window)
        images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Кнопка для загрузки изображений
        upload_btn = ctk.CTkButton(
            images_frame,
            text="Добавить изображения",
            command=lambda: self.add_image_to_compare(images_frame)
        )
        upload_btn.pack(pady=10)

    def add_image_to_compare(self, frame):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png"),
                ("Все файлы", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Создаем фрейм для изображения и результатов
                img_frame = ctk.CTkFrame(frame)
                img_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
                
                # Загружаем и отображаем изображение
                img = Image.open(file_path)
                img = self.resize_image(img, (200, 200))
                img_tk = ImageTk.PhotoImage(img)
                
                img_label = ctk.CTkLabel(
                    img_frame,
                    image=img_tk,
                    text=""
                )
                img_label.image = img_tk
                img_label.pack(pady=5)
                
                # Классифицируем изображение
                if self.model_loaded:
                    img_tensor = image.load_img(file_path, target_size=(224, 224))
                    x = image.img_to_array(img_tensor)
                    x = np.expand_dims(x, axis=0) / 255.0
                    
                    prediction = self.model.predict(x, verbose=0)[0]
                    idx = np.argmax(prediction)
                    confidence = float(np.max(prediction)) * 100
                    predicted_class = self.class_labels[list(self.class_labels.keys())[idx]]
                    
                    # Отображаем результаты
                    result_label = ctk.CTkLabel(
                        img_frame,
                        text=f"Класс: {predicted_class}\nУверенность: {confidence:.1f}%",
                        font=ctk.CTkFont(size=12)
                    )
                    result_label.pack(pady=5)
                    
            except Exception as e:
                messagebox.showerror(
                    "Ошибка",
                    f"Не удалось обработать изображение:\n{str(e)}"
                )

if __name__ == "__main__":
    root = ctk.CTk()
    app = ChessClassifierApp(root)
    root.mainloop() 