import matplotlib.pyplot as plt
import numpy as np

# Данные по этапам и их пропорционально увеличенная продолжительность (исходя из общего срока 150 дней вместо 120)
stages = {
    'Этап 1': 15 * (150/120),
    'Этап 2': 10 * (150/120),
    'Этап 3': 10 * (150/120),
    'Этап 4': 15 * (150/120),
    'Этап 5': 20 * (150/120),
    'Этап 6': 20 * (150/120),
    'Этап 7': 20 * (150/120),
    'Этап 8': 10 * (150/120),
}

# Названия этапов для оси Y
stage_labels = list(stages.keys())[::-1] # Разворачиваем для отображения на графике снизу вверх

# Расчет начальных и конечных дней для каждого этапа
start_day = 0
task_starts = []
task_durations = []

for stage in stages.values():
    task_starts.append(start_day)
    task_durations.append(stage)
    start_day += stage

# Построение графика
fig, ax = plt.subplots(figsize=(12, 6))

# Полосы для каждого этапа
bars = ax.barh(stage_labels, task_durations, left=task_starts, color='skyblue', height=0.8)

# Настройка осей и заголовка
ax.set_xlabel('Дни')
ax.set_title('Календарный график проведения работ')
ax.set_xlim(0, 150) # Устанавливаем предел по оси X до 150 дней

# Добавление вертикальных линий сетки для удобства
ax.xaxis.grid(True, linestyle='--', alpha=0.6)

# Убираем рамку графика
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.show()