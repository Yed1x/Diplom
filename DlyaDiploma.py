"""
ТЕХНИЧЕСКОЕ ЗАДАНИЕ
на разработку программного решения для классификации изображений с использованием глубокого обучения
(в рамках преддипломной практики)

1. ОБЩИЕ СВЕДЕНИЯ

1.1 Полное наименование системы и её условное обозначение
Полное наименование: Программное решение для классификации изображений с использованием глубокого обучения
Условное обозначение: СКИГО (Система Классификации Изображений с Глубоким Обучением)

1.2 Основание для разработки
Настоящее техническое задание разработано в соответствии с:
- ГОСТ 34.602-2020 "Информационные технологии. Комплекс стандартов на автоматизированные системы. Техническое задание на создание автоматизированной системы"
- Учебным планом специальности
- Заданием на преддипломную практику

1.3 Наименование заказчика и разработчика
Заказчик: Частное образовательное учреждение высшего образования «Московский университет имени С.Ю. Витте»
Разработчик: Студент группы [номер группы] [ФИО]

1.4 Цель и назначение разработки
Целью разработки является создание программного решения для автоматической классификации изображений с использованием современных методов глубокого обучения. Система предназначена для демонстрации практического применения знаний в области машинного обучения и разработки программного обеспечения.

2. ТРЕБОВАНИЯ К СИСТЕМЕ

2.1 Требования к функциональным характеристикам
2.1.1 Требования к составу выполняемых функций
Система должна обеспечивать:
- Загрузку и предобработку изображений
- Автоматическую классификацию изображений
- Визуализацию результатов классификации
- Сохранение истории классификаций

2.1.2 Требования к организации входных данных
- Поддерживаемые форматы: JPG, PNG, JPEG
- Максимальный размер файла: 5 МБ
- Размер обрабатываемых изображений: 224x224 пикселей

2.1.3 Требования к организации выходных данных
- Результаты классификации в формате JSON
- Визуализация результатов с указанием вероятности
- Статистические отчеты по классификации

2.2 Требования к надежности
- Точность классификации: не менее 85%
- Время обработки одного изображения: не более 1 секунды
- Устойчивость к различным форматам входных данных

2.3 Требования к составу и параметрам технических средств
2.3.1 Требования к аппаратному обеспечению
- Процессор: поддерживающий инструкции AVX2
- Оперативная память: не менее 8 ГБ
- Видеокарта: с поддержкой CUDA (для ускорения вычислений)
- Свободное место на диске: не менее 15 ГБ

2.3.2 Требования к программному обеспечению
- Операционная система: Windows 10/11 или Linux
- Python 3.8 или выше
- TensorFlow 2.x
- CUDA Toolkit (для работы с GPU)

2.4 Требования к информационной и программной совместимости
- Совместимость с веб-браузерами: Chrome, Firefox, Edge
- Поддержка REST API для интеграции с другими системами
- Совместимость с форматами данных JSON, CSV

3. СОСТАВ И СОДЕРЖАНИЕ РАБОТ ПО СОЗДАНИЮ СИСТЕМЫ

3.1 Этапы создания системы
1. Подготовительный этап:
   - Анализ требований
   - Выбор технологий и инструментов
   - Подготовка среды разработки

2. Разработка:
   - Реализация модели EfficientNetB0
   - Разработка системы аугментации данных
   - Создание веб-интерфейса

3. Тестирование:
   - Проверка точности классификации
   - Тестирование производительности
   - Проверка пользовательского интерфейса

4. Документирование:
   - Подготовка технической документации
   - Написание руководства пользователя
   - Оформление отчета по практике

3.2 Сроки выполнения работ
Начало работ: 01.02.2025
Окончание работ: 20.05.2025

4. ТРЕБОВАНИЯ К ДОКУМЕНТАЦИИ

4.1 Состав документации
- Техническое задание
- Отчет по преддипломной практике
- Руководство пользователя
- Исходный код с комментариями
- Презентация проекта

4.2 Требования к оформлению документации
- Соответствие ГОСТ 34.602-2020
- Наличие всех необходимых разделов
- Четкость и понятность изложения
- Актуальность информации

5. ТЕХНИКО-ЭКОНОМИЧЕСКИЕ ПОКАЗАТЕЛИ

5.1 Ожидаемые результаты
- Достижение точности классификации не менее 85%
- Создание рабочего прототипа системы
- Демонстрация практического применения знаний

5.2 Экономическая эффективность
- Возможность использования в учебном процессе
- Потенциал для дальнейшего развития
- Опыт практического применения технологий

6. ПОРЯДОК КОНТРОЛЯ И ПРИЕМКИ СИСТЕМЫ

6.1 Виды испытаний
- Функциональное тестирование
- Проверка точности классификации
- Тестирование пользовательского интерфейса

6.2 Общие требования к приемке работ
- Соответствие требованиям ТЗ
- Наличие полной документации
- Успешное прохождение всех тестов
- Достижение требуемых показателей

7. ПРИЛОЖЕНИЯ

7.1 Примеры изображений для тестирования
7.2 Описание форматов данных
7.3 Примеры отчетов о классификации
"""

2.2	Формирование набора данных

def create_data_generators(train_dir, val_dir):
    """
    Создает генераторы данных для обучения и валидации
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    



    ВЫБОР МОДЕЛИ 2.3



def create_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model




    2.4	Процесс обучения модели

   history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
