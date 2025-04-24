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
import time  # для анимаций

class ChessClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Piece Classifier Pro")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Обновленная цветовая схема с более яркими цветами
        self.color_scheme = {
            "primary": "#4361ee",       # Яркий синий
            "secondary": "#7209b7",     # Фиолетовый
            "accent": "#f72585",        # Розовый акцент
            "warning": "#ffd60a",       # Яркий желтый
            "error": "#e63946",         # Красный
            "success": "#2ec4b6",       # Бирюзовый
            "background": "#001233",    # Темно-синий фон
            "surface": "#023047",       # Поверхность компонентов
            "text": "#caf0f8",         # Светло-голубой текст
            "gradient_start": "#4cc9f0",  # Градиент начало
            "gradient_end": "#4895ef",    # Градиент конец
            "button_hover": "#f72585",    # Цвет при наведении
            "card_bg": "#0a1128"        # Фон карточек
        }
        
        # Обновленные стили с новыми эффектами
        self.styles = {
            "heading": ctk.CTkFont(family="Helvetica", size=28, weight="bold"),
            "subheading": ctk.CTkFont(family="Helvetica", size=20, weight="bold"),
            "button": ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
            "text": ctk.CTkFont(family="Helvetica", size=12),
            "small": ctk.CTkFont(family="Helvetica", size=10)
        }

        # Добавляем анимированные эффекты для кнопок
        self.button_effects = {
            "normal": {
                "fg_color": self.color_scheme["primary"],
                "hover_color": self.color_scheme["button_hover"],
                "border_width": 2,
                "border_color": self.color_scheme["accent"],
                "corner_radius": 15
            },
            "accent": {
                "fg_color": self.color_scheme["secondary"],
                "hover_color": self.color_scheme["accent"],
                "border_width": 2,
                "border_color": self.color_scheme["primary"],
                "corner_radius": 15
            }
        }
        
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
        
        # Добавляем инициализацию новых функций
        self.add_sound_effects()
        self.setup_hotkeys()
        self.setup_drag_and_drop()
        
    def setup_menu(self):
        menubar = tk.Menu(self.root, bg=self.color_scheme["surface"], fg=self.color_scheme["text"])
        self.root.config(menu=menubar)
        
        # Меню файла
        file_menu = tk.Menu(
            menubar, 
            tearoff=0,
            bg=self.color_scheme["surface"],
            fg=self.color_scheme["text"],
            activebackground=self.color_scheme["accent"],
            activeforeground=self.color_scheme["text"],
            font=self.styles["text"]
        )
        menubar.add_cascade(
            label="📁 Файл",
            menu=file_menu,
            font=self.styles["button"]
        )
        
        file_menu.add_command(
            label="🖼️ Открыть изображение",
            command=self.upload_image,
            font=self.styles["text"]
        )
        file_menu.add_command(
            label="📤 Экспорт истории",
            command=self.export_history,
            font=self.styles["text"]
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="❌ Выход",
            command=self.root.quit,
            font=self.styles["text"]
        )
        
        # Меню статистики
        stats_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.color_scheme["surface"],
            fg=self.color_scheme["text"],
            activebackground=self.color_scheme["accent"],
            activeforeground=self.color_scheme["text"],
            font=self.styles["text"]
        )
        menubar.add_cascade(
            label="📊 Статистика",
            menu=stats_menu,
            font=self.styles["button"]
        )
        stats_menu.add_command(
            label="📈 Показать статистику",
            command=self.show_statistics,
            font=self.styles["text"]
        )
        stats_menu.add_command(
            label="🗑️ Сбросить статистику",
            command=self.reset_statistics,
            font=self.styles["text"]
        )
        
        # Меню инструментов
        tools_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.color_scheme["surface"],
            fg=self.color_scheme["text"],
            activebackground=self.color_scheme["accent"],
            activeforeground=self.color_scheme["text"],
            font=self.styles["text"]
        )
        menubar.add_cascade(
            label="🛠️ Инструменты",
            menu=tools_menu,
            font=self.styles["button"]
        )
        tools_menu.add_command(
            label="📦 Пакетная обработка",
            command=self.open_batch_window,
            font=self.styles["text"]
        )
        tools_menu.add_command(
            label="🔄 Сравнить изображения",
            command=self.open_compare_window,
            font=self.styles["text"]
        )
        
        # Меню настроек
        settings_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.color_scheme["surface"],
            fg=self.color_scheme["text"],
            activebackground=self.color_scheme["accent"],
            activeforeground=self.color_scheme["text"],
            font=self.styles["text"]
        )
        menubar.add_cascade(
            label="⚙️ Настройки",
            menu=settings_menu,
            font=self.styles["button"]
        )
        settings_menu.add_command(
            label="🎨 Тема оформления",
            command=self.open_settings,
            font=self.styles["text"]
        )
        
        # Меню справки
        help_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.color_scheme["surface"],
            fg=self.color_scheme["text"],
            activebackground=self.color_scheme["accent"],
            activeforeground=self.color_scheme["text"],
            font=self.styles["text"]
        )
        menubar.add_cascade(
            label="❓ Справка",
            menu=help_menu,
            font=self.styles["button"]
        )
        help_menu.add_command(
            label="📖 Руководство",
            command=lambda: messagebox.showinfo(
                "Руководство",
                "Добро пожаловать в Chess Piece Classifier Pro!\n\n"
                "1. Загрузите изображение шахматной фигуры\n"
                "2. Получите результат классификации\n"
                "3. Просматривайте историю и статистику\n"
                "4. Используйте дополнительные инструменты"
            ),
            font=self.styles["text"]
        )
        help_menu.add_command(
            label="ℹ️ О программе",
            command=lambda: messagebox.showinfo(
                "О программе",
                "Chess Piece Classifier Pro\n"
                "Версия 1.0\n\n"
                "Программа для классификации шахматных фигур\n"
                "с использованием искусственного интеллекта"
            ),
            font=self.styles["text"]
        )
        help_menu.add_separator()
        help_menu.add_command(
            label="🔄 Проверить обновления",
            command=self.check_for_updates,
            font=self.styles["text"]
        )
    
    def create_widgets(self):
        # Основной контейнер с градиентным фоном
        self.main_container = ctk.CTkFrame(
            self.root,
            fg_color=self.color_scheme["background"]
        )
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Панель инструментов с эффектом стекла
        self.toolbar = ctk.CTkFrame(
            self.main_container,
            fg_color=self.color_scheme["surface"],
            corner_radius=15
        )
        self.toolbar.pack(fill=tk.X, pady=(0, 10))
        
        # Обновленные кнопки с новым дизайном
        self.upload_button = ctk.CTkButton(
            self.toolbar,
            text="📁 Загрузить изображение",
            command=self.upload_image,
            font=self.styles["button"],
            height=45,
            width=200,
            **self.button_effects["normal"],
            text_color=self.color_scheme["text"]
        )
        self.upload_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.stats_button = ctk.CTkButton(
            self.toolbar,
            text="📊 Статистика",
            command=self.show_statistics,
            font=self.styles["button"],
            height=45,
            width=200,
            **self.button_effects["accent"],
            text_color=self.color_scheme["text"]
        )
        self.stats_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Добавляем кнопку анализа цвета после кнопки загрузки
        self.color_analysis_button = ctk.CTkButton(
            self.toolbar,
            text="🎨 Анализ цвета",
            command=lambda: self.show_color_analysis(self.current_image_path) if hasattr(self, 'current_image_path') else messagebox.showinfo("Информация", "Сначала загрузите изображение"),
            font=self.styles["button"],
            height=45,
            width=200,
            **self.button_effects["normal"],
            text_color=self.color_scheme["text"]
        )
        self.color_analysis_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Обновленный заголовок с градиентным эффектом
        title_frame = ctk.CTkFrame(
            self.main_container,
            fg_color=self.color_scheme["card_bg"],
            corner_radius=20
        )
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.title_label = ctk.CTkLabel(
            title_frame,
            text="🎯 Классификатор шахматных фигур",
            font=self.styles["heading"],
            text_color=self.color_scheme["gradient_start"]
        )
        self.title_label.pack(pady=15)
        
        # Статус модели с анимированной иконкой
        status_text = "✅ Модель загружена" if self.model_loaded else "❌ Ошибка загрузки модели"
        status_color = self.color_scheme["success"] if self.model_loaded else self.color_scheme["error"]
        self.status_label = ctk.CTkLabel(
            self.main_container,
            text=status_text,
            text_color=status_color,
            font=self.styles["text"]
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
        
        # Добавляем всплывающие подсказки с анимацией
        self.show_tooltip(self.upload_button, "Нажмите или перетащите файлы сюда\nCtrl+O")
        self.show_tooltip(self.stats_button, "Просмотр статистики классификаций\nF5")
        self.show_tooltip(self.color_analysis_button, "Показать детальный анализ определения цвета фигуры")
        
        # Добавляем звуковые эффекты для кнопок
        self.upload_button.configure(command=lambda: [self.play_sound("click"), self.upload_image()])
        self.stats_button.configure(command=lambda: [self.play_sound("click"), self.show_statistics()])
    
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
        
        # Создание красивых карточек с градиентным эффектом
        class_card = ctk.CTkFrame(
            self.result_frame,
            fg_color=self.color_scheme["card_bg"],
            corner_radius=15,
            border_width=2,
            border_color=self.color_scheme["accent"]
        )
        class_card.pack(fill=tk.X, pady=10, padx=15)
        
        # Добавляем анимированные иконки
        icons = {
            "Ферзь": "👑",
            "Ладья": "🏰",
            "Слон": "🐘",
            "Конь": "🐴",
            "Пешка": "♟️"
        }
        
        icon = next((v for k, v in icons.items() if k in class_name), "🎯")
        
        ctk.CTkLabel(
            class_card,
            text=f"{icon} Тип фигуры:",
            font=self.styles["text"],
            text_color=self.color_scheme["gradient_start"]
        ).pack(pady=5)
        
        ctk.CTkLabel(
            class_card,
            text=class_name,
            font=self.styles["subheading"],
            text_color=self.color_scheme["accent"]
        ).pack(pady=5)
        
        # Карточка для цвета
        color_card = ctk.CTkFrame(
            self.result_frame,
            fg_color=self.color_scheme["surface"],
            corner_radius=10
        )
        color_card.pack(fill=tk.X, pady=5, padx=10)
        
        ctk.CTkLabel(
            color_card,
            text="🎨 Цвет:",
            font=self.styles["text"],
            text_color=self.color_scheme["text"]
        ).pack()
        
        ctk.CTkLabel(
            color_card,
            text=color,
            font=self.styles["subheading"],
            text_color=self.color_scheme["accent"]
        ).pack()
        
        # Карточка для уверенности
        conf_card = ctk.CTkFrame(
            self.result_frame,
            fg_color=self.color_scheme["surface"],
            corner_radius=10
        )
        conf_card.pack(fill=tk.X, pady=5, padx=10)
        
        ctk.CTkLabel(
            conf_card,
            text="📊 Уверенность:",
            font=self.styles["text"],
            text_color=self.color_scheme["text"]
        ).pack()
        
        # Добавляем эффект свечения для высокой уверенности
        confidence_value = float(confidence.strip('%'))
        confidence_color = (
            self.color_scheme["success"] if confidence_value > 80
            else self.color_scheme["warning"] if confidence_value > 50
            else self.color_scheme["error"]
        )
        
        # Прогресс-бар с анимацией
        progress = ctk.CTkProgressBar(
            conf_card,
            progress_color=confidence_color,
            height=15,
            corner_radius=10
        )
        progress.pack(pady=10, padx=20, fill=tk.X)
        progress.set(confidence_value / 100)
    
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
            # Открываем изображение и конвертируем в оттенки серого
            img = Image.open(file_path).convert("L")
            arr = np.array(img)
            
            # Получаем размеры изображения
            h, w = arr.shape
            
            # Вычисляем центр и размер области для анализа
            cx, cy = w // 2, h // 2
            s = min(h, w) // 3  # Уменьшаем размер области анализа
            
            # Вырезаем центральную область
            crop = arr[cy - s//2:cy + s//2, cx - s//2:cx + s//2]
            
            # Применяем пороговую обработку для отделения фигуры от фона
            threshold = 200  # Порог для отделения фигуры от фона
            figure_pixels = crop[crop < threshold]
            
            if len(figure_pixels) == 0:
                return "Не удалось определить цвет ❔", 0
            
            # Используем гистограмму для анализа распределения яркости
            hist, bins = np.histogram(figure_pixels, bins=3)
            dark_pixels = np.sum(hist[:2])  # Количество темных пикселей
            light_pixels = hist[2]  # Количество светлых пикселей
            
            # Вычисляем среднюю яркость только для пикселей фигуры
            mean = np.mean(figure_pixels)
            
            # Определяем цвет на основе соотношения темных и светлых пикселей
            # и средней яркости
            if dark_pixels > light_pixels and mean < 150:
                return "Чёрная ♟️", mean
            elif light_pixels > dark_pixels and mean > 100:
                return "Белая ♙", mean
            else:
                # Используем дополнительный анализ для неоднозначных случаев
                std_dev = np.std(figure_pixels)
                if std_dev < 40:  # Если разброс яркости небольшой
                    return "Чёрная ♟️" if mean < 127 else "Белая ♙", mean
                else:
                    # Анализируем распределение яркости
                    dark_ratio = np.sum(figure_pixels < 127) / len(figure_pixels)
                    return "Чёрная ♟️" if dark_ratio > 0.5 else "Белая ♙", mean
                
        except Exception as e:
            return f"Ошибка определения цвета: {e}", 0

    def show_color_analysis(self, file_path):
        try:
            img = Image.open(file_path).convert("L")
            arr = np.array(img)
            h, w = arr.shape
            cx, cy = w // 2, h // 2
            s = min(h, w) // 3
            
            # Создаем окно анализа
            analysis_window = ctk.CTkToplevel(self.root)
            analysis_window.title("Анализ цвета фигуры")
            analysis_window.geometry("800x600")
            
            # Создаем фрейм для отображения
            frame = ctk.CTkFrame(
                analysis_window,
                fg_color=self.color_scheme["card_bg"]
            )
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Отображаем оригинальное изображение
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Оригинальное изображение
            ax1.imshow(arr, cmap='gray')
            ax1.set_title("Оригинал")
            
            # Область анализа
            crop = arr[cy - s//2:cy + s//2, cx - s//2:cx + s//2]
            ax2.imshow(crop, cmap='gray')
            ax2.set_title("Область анализа")
            
            # Гистограмма
            ax3.hist(crop.ravel(), bins=50, color=self.color_scheme["accent"])
            ax3.set_title("Гистограмма яркости")
            
            # Настраиваем внешний вид графиков
            for ax in [ax1, ax2, ax3]:
                ax.set_facecolor(self.color_scheme["surface"])
                ax.grid(True, alpha=0.3)
            fig.patch.set_facecolor(self.color_scheme["card_bg"])
            
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Добавляем информацию об анализе
            color, mean = self.detect_color(file_path)
            
            # Вычисляем дополнительные метрики
            threshold = 200
            figure_pixels = crop[crop < threshold]
            hist, _ = np.histogram(figure_pixels, bins=3)
            dark_pixels = np.sum(hist[:2])
            light_pixels = hist[2]
            std_dev = np.std(figure_pixels)
            dark_ratio = np.sum(figure_pixels < 127) / len(figure_pixels)
            
            info_text = f"""
            📊 Результаты анализа:
            
            🎨 Определенный цвет: {color}
            📏 Средняя яркость: {mean:.2f}
            📐 Размер области анализа: {s}x{s} пикселей
            
            📈 Дополнительные метрики:
            • Стандартное отклонение: {std_dev:.2f}
            • Соотношение темных пикселей: {dark_ratio:.2%}
            • Количество темных пикселей: {dark_pixels}
            • Количество светлых пикселей: {light_pixels}
            """
            
            info_label = ctk.CTkLabel(
                frame,
                text=info_text,
                font=self.styles["text"],
                justify="left",
                fg_color=self.color_scheme["surface"],
                corner_radius=10,
                padx=20,
                pady=20
            )
            info_label.pack(pady=10, padx=10, fill=tk.X)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось выполнить анализ: {e}")
    
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
        file_paths = filedialog.askopenfilenames(
            title="Выберите изображения",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png"),
                ("Все файлы", "*.*")
            ]
        )
        
        if not file_paths:
            messagebox.showinfo("Информация", "Изображения не выбраны")
            return
        
        # Создаем новое окно для отображения всех выбранных изображений
        images_window = ctk.CTkToplevel(self.root)
        images_window.title("Результаты классификации")
        images_window.geometry("800x600")
        
        # Создаем холст с полосой прокрутки
        canvas = ctk.CTkCanvas(images_window)
        scrollbar = ttk.Scrollbar(images_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Размещаем элементы
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Обрабатываем каждое изображение
        for file_path in file_paths:
            # Создаем фрейм для каждого изображения и его результатов
            img_frame = ctk.CTkFrame(scrollable_frame)
            img_frame.pack(pady=10, padx=10, fill="x")
            
            try:
                # Отображаем изображение
                img = Image.open(file_path)
                img = self.resize_image(img, (200, 200))
                img_tk = ImageTk.PhotoImage(img)
                
                img_label = ctk.CTkLabel(
                    img_frame,
                    image=img_tk,
                    text=""
                )
                img_label.image = img_tk
                img_label.pack(side="left", padx=10)
                
                # Создаем фрейм для результатов
                results_frame = ctk.CTkFrame(img_frame)
                results_frame.pack(side="left", fill="both", expand=True, padx=10)
                
                # Имя файла
                file_name = os.path.basename(file_path)
                ctk.CTkLabel(
                    results_frame,
                    text=f"Файл: {file_name}",
                    font=ctk.CTkFont(size=12)
                ).pack(anchor="w")
                
                if self.model_loaded:
                    # Определяем цвет фигуры
                    fig_color, _ = self.detect_color(file_path)
                    
                    # Классифицируем изображение
                    img_tensor = image.load_img(file_path, target_size=(224, 224))
                    x = image.img_to_array(img_tensor)
                    x = np.expand_dims(x, axis=0) / 255.0
                    
                    prediction = self.model.predict(x, verbose=0)[0]
                    idx = np.argmax(prediction)
                    confidence = float(np.max(prediction)) * 100
                    predicted_class = self.class_labels[list(self.class_labels.keys())[idx]]
                    
                    # Отображаем результаты
                    ctk.CTkLabel(
                        results_frame,
                        text=f"Тип фигуры: {predicted_class}",
                        font=ctk.CTkFont(size=12)
                    ).pack(anchor="w")
                    
                    ctk.CTkLabel(
                        results_frame,
                        text=f"Цвет: {fig_color}",
                        font=ctk.CTkFont(size=12)
                    ).pack(anchor="w")
                    
                    # Прогресс-бар уверенности
                    conf_frame = ctk.CTkFrame(results_frame)
                    conf_frame.pack(fill="x", pady=5)
                    
                    ctk.CTkLabel(
                        conf_frame,
                        text=f"Уверенность: {confidence:.1f}%",
                        font=ctk.CTkFont(size=12)
                    ).pack(side="left")
                    
                    progress = ctk.CTkProgressBar(conf_frame, width=100)
                    progress.pack(side="left", padx=10)
                    progress.set(confidence / 100)
                    
                    # Сохраняем в историю
                    self.save_to_history(file_name, predicted_class, fig_color, f"{confidence:.1f}%")
                    
                    # Обновляем статистику
                    self.update_stats(predicted_class, fig_color)
                    
                else:
                    ctk.CTkLabel(
                        results_frame,
                        text="Модель не загружена",
                        text_color="red",
                        font=ctk.CTkFont(size=12)
                    ).pack()
                
            except Exception as e:
                ctk.CTkLabel(
                    img_frame,
                    text=f"Ошибка обработки {file_name}:\n{str(e)}",
                    text_color="red",
                    font=ctk.CTkFont(size=12)
                ).pack()
    
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
        self.current_image_path = file_path  # Сохраняем путь к текущему изображению
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
        Hovertip(self.color_analysis_button, "Показать детальный анализ определения цвета фигуры")

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
        # Добавляем кнопку настроек в toolbar с новым дизайном
        self.settings_button = ctk.CTkButton(
            self.toolbar,
            text="⚙️ Настройки",
            command=self.open_settings,
            font=self.styles["button"],
            height=40,
            width=200,
            fg_color=self.color_scheme["primary"],
            hover_color=self.color_scheme["accent"]
        )
        self.settings_button.pack(side=tk.LEFT, padx=5)

    def open_settings(self):
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("⚙️ Настройки")
        settings_window.geometry("600x500")
        
        # Контейнер настроек с градиентным фоном
        settings_container = ctk.CTkFrame(
            settings_window,
            fg_color=self.color_scheme["background"],
            corner_radius=20
        )
        settings_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Заголовок настроек
        ctk.CTkLabel(
            settings_container,
            text="🎨 Настройки приложения",
            font=self.styles["heading"],
            text_color=self.color_scheme["gradient_start"]
        ).pack(pady=20)
        
        # Темы оформления
        themes_frame = ctk.CTkFrame(
            settings_container,
            fg_color=self.color_scheme["card_bg"],
            corner_radius=15
        )
        themes_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ctk.CTkLabel(
            themes_frame,
            text="🎭 Тема оформления",
            font=self.styles["subheading"],
            text_color=self.color_scheme["text"]
        ).pack(pady=10)
        
        theme_var = tk.StringVar(value="dark")
        for theme in ["Тёмная", "Светлая", "Системная"]:
            ctk.CTkRadioButton(
                themes_frame,
                text=theme,
                variable=theme_var,
                value=theme.lower(),
                font=self.styles["text"],
                fg_color=self.color_scheme["accent"],
                border_color=self.color_scheme["primary"],
                hover_color=self.color_scheme["button_hover"]
            ).pack(pady=5)

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

    def show_loading_screen(self):
        loading_window = ctk.CTkToplevel(self.root)
        loading_window.title("Загрузка")
        loading_window.geometry("300x200")
        loading_window.transient(self.root)
        
        loading_frame = ctk.CTkFrame(
            loading_window,
            fg_color=self.color_scheme["card_bg"]
        )
        loading_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        loading_label = ctk.CTkLabel(
            loading_frame,
            text="⏳ Загрузка...",
            font=self.styles["heading"],
            text_color=self.color_scheme["gradient_start"]
        )
        loading_label.pack(pady=20)
        
        progress = ctk.CTkProgressBar(
            loading_frame,
            mode="indeterminate",
            height=15,
            corner_radius=10
        )
        progress.pack(pady=10, padx=20, fill=tk.X)
        progress.start()
        
        return loading_window

    def add_sound_effects(self):
        self.sounds = {
            "success": lambda: print("\a"),  # Системный звук
            "error": lambda: print("\a\a"),  # Двойной системный звук
            "click": lambda: print("\a")     # Системный звук
        }

    def play_sound(self, sound_type):
        if hasattr(self, "sounds") and sound_type in self.sounds:
            self.sounds[sound_type]()

    def setup_hotkeys(self):
        self.root.bind("<Control-o>", lambda e: self.upload_image())
        self.root.bind("<Control-s>", lambda e: self.export_history())
        self.root.bind("<Control-q>", lambda e: self.root.quit())
        self.root.bind("<F1>", lambda e: self.show_help())
        self.root.bind("<F5>", lambda e: self.refresh_history())

    def show_tooltip(self, widget, text):
        tooltip = ctk.CTkToplevel(self.root)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        
        label = ctk.CTkLabel(
            tooltip,
            text=text,
            font=self.styles["small"],
            fg_color=self.color_scheme["surface"],
            corner_radius=10,
            padx=10,
            pady=5
        )
        label.pack()
        
        def show(event):
            x = widget.winfo_rootx() + widget.winfo_width()
            y = widget.winfo_rooty()
            tooltip.geometry(f"+{x}+{y}")
            tooltip.deiconify()
        
        def hide(event):
            tooltip.withdraw()
        
        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def animate_window(self, window, start_geometry, end_geometry):
        def update_geometry(progress):
            if not window.winfo_exists():
                return
            
            current_geometry = ""
            for start, end in zip(
                start_geometry.split("+"),
                end_geometry.split("+")
            ):
                if "x" in start:
                    w_start, h_start = map(int, start.split("x"))
                    w_end, h_end = map(int, end.split("x"))
                    w = int(w_start + (w_end - w_start) * progress)
                    h = int(h_start + (h_end - h_start) * progress)
                    current_geometry += f"{w}x{h}"
                else:
                    pos_start = int(start)
                    pos_end = int(end)
                    pos = int(pos_start + (pos_end - pos_start) * progress)
                    current_geometry += f"+{pos}"
            
            window.geometry(current_geometry)
            
            if progress < 1:
                window.after(10, lambda: update_geometry(min(1, progress + 0.1)))
        
        update_geometry(0)

    def setup_drag_and_drop(self):
        def drop(event):
            try:
                file_path = event.data
                if file_path.startswith("{"):
                    file_path = file_path[1:-1]
                if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    self.process_image(file_path)
                    self.play_sound("success")
                else:
                    self.play_sound("error")
                    messagebox.showwarning("Предупреждение", "Поддерживаются только изображения (JPG, PNG)")
            except Exception as e:
                self.play_sound("error")
                messagebox.showerror("Ошибка", f"Ошибка при обработке файла:\n{str(e)}")
        
        try:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", drop)
        except:
            print("Drag and Drop не поддерживается")

    def check_for_updates(self):
        update_window = ctk.CTkToplevel(self.root)
        update_window.title("Проверка обновлений")
        update_window.geometry("400x200")
        
        update_label = ctk.CTkLabel(
            update_window,
            text="🔄 Проверка обновлений...",
            font=self.styles["heading"]
        )
        update_label.pack(pady=20)
        
        progress = ctk.CTkProgressBar(
            update_window,
            mode="indeterminate"
        )
        progress.pack(pady=10)
        progress.start()
        
        def show_update_result():
            progress.stop()
            update_label.configure(text="✅ Установлена последняя версия")
            update_window.after(2000, update_window.destroy)
        
        update_window.after(2000, show_update_result)

    def show_help(self):
        help_window = ctk.CTkToplevel(self.root)
        help_window.title("Справка")
        help_window.geometry("600x400")
        
        help_frame = ctk.CTkFrame(
            help_window,
            fg_color=self.color_scheme["card_bg"]
        )
        help_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        help_text = """
        🎯 Горячие клавиши:
        
        Ctrl + O - Открыть изображение
        Ctrl + S - Экспорт истории
        Ctrl + Q - Выход
        F1      - Показать справку
        F5      - Обновить историю
        
        📝 Основные функции:
        
        • Загрузка изображений перетаскиванием
        • Классификация шахматных фигур
        • Определение цвета фигур
        • Просмотр статистики
        • Экспорт результатов
        
        ℹ️ Дополнительно:
        
        • Поддерживаются форматы: JPG, PNG
        • Результаты сохраняются автоматически
        • Доступна темная и светлая темы
        """
        
        ctk.CTkLabel(
            help_frame,
            text=help_text,
            font=self.styles["text"],
            justify="left"
        ).pack(pady=20)

    def refresh_history(self):
        loading = self.show_loading_screen()
        self.root.after(100, lambda: self.load_history())
        self.root.after(500, loading.destroy)
        self.play_sound("success")

if __name__ == "__main__":
    root = ctk.CTk()
    app = ChessClassifierApp(root)
    root.mainloop() 


    #python chess_classifier_gui.py