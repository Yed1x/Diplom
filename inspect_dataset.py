import json
import matplotlib.pyplot as plt

# Загрузка истории из JSON
with open('stats.json', 'r') as f:
    history = json.load(f)

# Построение графика accuracy
plt.figure(figsize=(8, 5))
plt.plot(history['accuracy'], label='Train accuracy')
plt.plot(history['val_accuracy'], label='Validation accuracy')
plt.title('График точности обучения')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)
plt.savefig('training_accuracy.png')
plt.show()

# Построение графика loss
plt.figure(figsize=(8, 5))
plt.plot(history['loss'], label='Train loss')
plt.plot(history['val_loss'], label='Validation loss')
plt.title('График функции потерь')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()