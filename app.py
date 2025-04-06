# app.py — расширенный, премиум-уровень интерфейс
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Chess Classifier Pro", page_icon="♟️", layout="centered")
st.title("🧠♟️ Определение шахматной фигуры — Pro-версия")

# Модель и классы
model = load_model("chess_model_updated.keras")
class_labels = {
    'Queen-Resized': 'Ферзь 👑',
    'Rook-resize': 'Ладья 🏰',
    'bishop_resized': 'Слон 🐘',
    'knight-resize': 'Конь 🐴',
    'pawn_resized': 'Пешка 🧍‍♂️'
}
log_file = "predictions_log.csv"

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

# 🧾 История
if os.path.exists(log_file):
    try:
        df_log = pd.read_csv(log_file, encoding='utf-8', errors='replace')
        if "Цвет" not in df_log.columns:
            df_log["Цвет"] = "—"
        st.subheader("📋 История предсказаний")
        st.dataframe(df_log.tail(5), use_container_width=True)
    except:
        st.warning("⚠️ Лог повреждён или не читается")

# 📤 Загрузка и предсказание
uploaded_file = st.file_uploader("Загрузите изображение фигуры", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="🖼️ Загружено", use_container_width=True)

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

    st.success(f"🟢 Модель определила: **{predicted_class}**, цвет: **{fig_color}**, уверенность: **{confidence:.2f}%**")

    # 📊 График уверенности
    st.subheader("📊 Уверенность по классам")
    fig, ax = plt.subplots()
    ax.bar(class_labels.values(), prediction * 100)
    ax.set_ylabel('%')
    ax.set_title('Уверенность модели')
    st.pyplot(fig)

    # 💾 Сохраняем лог
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df = pd.DataFrame([{
        "Время": now,
        "Файл": uploaded_file.name,
        "Класс": predicted_class,
        "Цвет": fig_color,
        "Уверенность": round(confidence, 2)
    }])

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
