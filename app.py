import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import csv

# Настройка темы и цветов
st.set_page_config(page_title="Chess Classifier Pro", page_icon="♟️", layout="centered")

# Кастомные стили
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTitle {
        color: #2c3e50;
    }
    .stSubheader {
        color: #34495e;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
    }
    .stInfo {
        background-color: #cce5ff;
        color: #004085;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.facecolor'] = '#f5f5f5'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#2c3e50'
plt.rcParams['text.color'] = '#2c3e50'
plt.rcParams['axes.labelcolor'] = '#2c3e50'

print(plt.style.available)

st.title("🧠♟️ Определение шахматной фигуры — Pro-версия")

# Модель и классы
model = load_model("final_model.h5")
class_labels = {
    'bishop': 'Слон 🐘',
    'knight': 'Конь 🐴',
    'pawn': 'Пешка 🧍‍♂️',
    'queen': 'Ферзь 👑',
    'rook': 'Ладья 🏰'
}
log_file = "predictions_log.csv"

# Автоматическая инициализация лога, если файл отсутствует или повреждён
if not os.path.exists(log_file):
    with open(log_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Файл", "Класс", "Цвет", "Уверенность"])
else:
    # Проверяем корректность заголовков
    try:
        with open(log_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            headers = next(reader)
            if headers != ["Файл", "Класс", "Цвет", "Уверенность"]:
                raise ValueError
    except Exception:
        with open(log_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Файл", "Класс", "Цвет", "Уверенность"])

# 🎨 Определение цвета и вывод кропа
def detect_color_preview(uploaded_file):
    st.write("✅ detect_color_preview ВЫЗВАНА")
    try:
        img = Image.open(uploaded_file).convert("L")
        st.write("🖼️ Файл открыт через PIL, размер:", img.size)
        arr = np.array(img)
        h, w = arr.shape
        cx, cy = w // 2, h // 2
        s = min(h, w) // 2
        crop = arr[cy - s//2:cy + s//2, cx - s//2:cx + s//2]
        crop_preview = Image.fromarray(crop)
        crop_valid = crop[crop < 240]

        if len(crop_valid) == 0:
            st.warning("⚠️ Центр изображения слишком светлый. Цвет определить не удалось.")
            return "Не удалось определить цвет ❔", 0, crop_preview

        mean = np.mean(crop_valid)
        color = "Чёрная ♟️" if mean < 127 else "Белая ♙"
        return color, mean, crop_preview
    except Exception as e:
        st.error(f"❌ Не удалось открыть изображение: {e}")
        return "Ошибка ❌", 0, None

# 🧾 История с фильтрацией
if os.path.exists(log_file):
    try:
        # Пробуем разные кодировки
        encodings = ['utf-8-sig', 'utf-8', 'cp1251']
        df_log = None
        
        for encoding in encodings:
            try:
                df_log = pd.read_csv(log_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df_log is None:
            raise ValueError("Не удалось прочитать файл ни с одной кодировкой")
            
        # Проверяем наличие всех необходимых колонок
        required_columns = ["Файл", "Класс", "Цвет", "Уверенность"]
        missing_columns = [col for col in required_columns if col not in df_log.columns]
        
        if missing_columns:
            # Если каких-то колонок нет, добавляем их
            for col in missing_columns:
                df_log[col] = "—"
        
        # Фильтры
        st.subheader("📋 История предсказаний")
        color_filter = st.selectbox("Фильтр по цвету", options=["Все", "Чёрная", "Белая"])
        class_filter = st.selectbox("Фильтр по классу", options=["Все"] + list(class_labels.values()))
        
        # Применяем фильтры
        if color_filter != "Все":
            df_log = df_log[df_log["Цвет"].str.contains(color_filter, na=False)]
        
        if class_filter != "Все":
            df_log = df_log[df_log["Класс"] == class_filter]
        
        if not df_log.empty:
            st.dataframe(df_log.tail(5), use_container_width=True)
        else:
            st.info("История пуста")
            
    except Exception as e:
        st.warning(f"⚠️ Проблема с чтением лога: {str(e)}")
        # Создаем новый файл лога
        with open(log_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Файл", "Класс", "Цвет", "Уверенность"])

# 📤 Загрузка и предсказание для нескольких изображений
uploaded_files = st.file_uploader("Загрузите изображения фигур", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    predictions = []
    
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption=f"🖼️ Загружено: {uploaded_file.name}", use_container_width=True)

        st.write("🧪 Пытаемся определить цвет через detect_color_preview")
        fig_color, brightness, center_crop = detect_color_preview(uploaded_file)

        if center_crop:
            st.image(center_crop, caption=f"🔍 Центр для анализа цвета (яркость: {brightness:.2f})", use_container_width=True)

        color_emoji = "⚫️" if "Чёрная" in fig_color else "⚪️" if "Белая" in fig_color else "❔"
        st.markdown(f"### {color_emoji} Цвет фигуры: **{fig_color}**")

        img = image.load_img(uploaded_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        prediction = model.predict(x)[0]
        idx = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        predicted_class = class_labels[list(class_labels.keys())[idx]]

        # Топ-3 вероятности
        top3_idx = np.argsort(prediction)[::-1][:3]
        top3_labels = [list(class_labels.values())[i] for i in top3_idx]
        top3_probs = [prediction[i]*100 for i in top3_idx]
        st.info("Топ-3 вероятности:")
        for label, prob in zip(top3_labels, top3_probs):
            st.write(f"{label}: {prob:.2f}%")

        st.success(f"🟢 Модель определила: **{predicted_class}**, цвет: **{fig_color}**, уверенность: **{confidence:.2f}%**")

        # 📊 График уверенности
        st.subheader("📊 Уверенность по классам")
        fig, ax = plt.subplots()
        ax.bar(class_labels.values(), prediction * 100)
        ax.set_ylabel('%')
        ax.set_title('Уверенность модели')
        st.pyplot(fig)

        # Добавляем результаты в список
        predictions.append({
            "Файл": uploaded_file.name,
            "Класс": predicted_class,
            "Цвет": fig_color,
            "Уверенность": f"{confidence:.2f}%"
        })

    # Показываем результаты всех изображений
    st.subheader("📋 Результаты сравнения изображений:")
    for pred in predictions:
        st.write(f"Файл: {pred['Файл']} — **{pred['Класс']}** (Цвет: {pred['Цвет']}), Уверенность: {pred['Уверенность']}")

    # 💾 Сохраняем лог
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df = pd.DataFrame(predictions)
    
    if os.path.exists(log_file):
        try:
            current = pd.read_csv(log_file, encoding='utf-8')
            if set(current.columns) == set(log_df.columns):
                log_df.to_csv(log_file, mode='a', index=False, header=False, encoding='utf-8-sig')
            else:
                log_df.to_csv(log_file, index=False, encoding='utf-8-sig')
        except:
            log_df.to_csv(log_file, index=False, encoding='utf-8-sig')
    else:
        log_df.to_csv(log_file, index=False, encoding='utf-8-sig')

# Запуск вот такой # python -m streamlit run app.py
