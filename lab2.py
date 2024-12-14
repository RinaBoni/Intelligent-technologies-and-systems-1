import numpy as np


# Функция активации: сигмоид
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Производная функции активации: сигмоид
def sigmoid_derivative(x):
    return x * (1 - x)


# Класс многослойного персептрона
class MultilayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        # Инициализация весов случайными значениями
        self.W1 = np.random.randn(input_size, hidden_size)  # Веса для первого слоя
        self.W2 = np.random.randn(hidden_size, output_size)  # Веса для второго слоя

        # Инициализация смещений (bias) нулями
        self.bias1 = np.zeros((1, hidden_size))  # Смещение для первого слоя
        self.bias2 = np.zeros((1, output_size))  # Смещение для второго слоя

    def forward(self, X):
        # Прямое распространение: вычисление выходов слоев
        self.layer1 = sigmoid(np.dot(X, self.W1) + self.bias1)  # Выход первого слоя
        self.layer2 = sigmoid(np.dot(self.layer1, self.W2) + self.bias2)  # Выход второго слоя
        return self.layer2  # Возвращаем выход сети

    def backward(self, X, y, lr):
        # Обратное распространение ошибки
        output_error = y - self.layer2  # Ошибка на выходе
        delta2 = output_error * sigmoid_derivative(self.layer2)  # Вычисляем дельту для второго слоя

        hidden_error = np.dot(delta2, self.W2.T)  # Ошибка для скрытого слоя
        delta1 = hidden_error * sigmoid_derivative(self.layer1)  # Вычисляем дельту для первого слоя

        # Обновление весов и смещений
        self.W2 += lr * np.dot(self.layer1.T, delta2)  # Обновление весов второго слоя
        self.W1 += lr * np.dot(X.T, delta1)  # Обновление весов первого слоя

        self.bias2 += lr * np.sum(delta2, axis=0, keepdims=True)  # Обновление смещения второго слоя
        self.bias1 += lr * np.sum(delta1, axis=0, keepdims=True)  # Обновление смещения первого слоя

    def train(self, X, y, epochs, lr):
        # Обучение модели
        for epoch in range(epochs):
            output = self.forward(X)  # Прямое распространение
            self.backward(X, y, lr)  # Обратное распространение


# Генерация простых данных для обучения
# Входные данные (X) и соответствующие метки (y)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])  # Данные для логической операции AND
y = np.array([[0], [0], [0], [1]])  # Ожидаемые выходы

# Создание экземпляра многослойного персептрона
input_size = 2  # Количество входов
hidden_size = 2  # Количество нейронов в скрытом слое
output_size = 1  # Количество выходов

mlp = MultilayerPerceptron(input_size, hidden_size, output_size)

# Обучение модели
epochs = 10000  # Количество эпох
learning_rate = 0.1  # Скорость обучения
mlp.train(X, y, epochs, learning_rate)

# Тестирование модели
print("Тестирование модели:")
for i in range(len(X)):
    output = mlp.forward(X[i].reshape(1, -1))  # Прямое распространение для каждого входа
    print(f"Вход: {X[i]} -> Выход: {output} (ожидаемый: {y[i]})")