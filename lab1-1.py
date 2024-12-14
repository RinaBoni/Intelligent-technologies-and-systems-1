import random
import datetime

# Набор возможных символов для генерации строк
geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."
# Целевая строка, которую мы хотим получить
target = "you are the best"
# target = "Hello World!"


def generate_parent(length):
    """Создает случайную строку (родителя) заданной длины, выбирая символы из geneSet."""
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))  # Определяем размер выборки, чтобы не превышать длину целевой строки
        genes.extend(random.sample(geneSet, sampleSize))     # Добавляем случайные символы из geneSet
    return ''.join(genes)


def get_fitness(guess):
    """Вычисляет, насколько близка предполагаемая строка к целевой строке,
       подсчитывая количество символов, которые совпадают на тех же позициях."""
    return sum(1 for expected, actual in zip(target, guess)  # Сравниваем каждый символ предполагаемой строки с целевой строкой
               if expected == actual)


def mutate(parent):
    """Создает новую строку (потомка), случайно изменяя один символ
       родительской строки. Она выбирает случайный индекс и заменяет символ
       в этом индексе на новый символ из geneSet."""
    index = random.randrange(0, len(parent))        # Выбираем случайный индекс для изменения символа
    childGenes = list(parent)                       # Преобразуем строку в список для изменения
    newGene, alternate = random.sample(geneSet, 2)  # Выбираем два случайных символа из geneSet
    # Заменяем символ в выбранном индексе на новый символ
    childGenes[index] = alternate \
        if newGene == childGenes[index] \
        else newGene
    return ''.join(childGenes)                      # Возвращаем измененную строку


def display(guess):
    """Выводит текущую строку, ее оценку приспособленности и время,
       прошедшее с начала алгоритма"""
    timeDiff = datetime.datetime.now() - startTime  # Вычисляем время выполнения
    fitness = get_fitness(guess)                    # Получаем приспособленность текущей строки
    # Выводим строку, ее приспособленность и время выполнения
    print("{0}\t{1}\t{2}".format(guess, fitness, str(timeDiff)))


# Инициализируем генератор случайных чисел и времени начала
random.seed()
startTime = datetime.datetime.now()

bestParent = generate_parent(len(target))   # Генерируем случайную родительскую строку
bestFitness = get_fitness(bestParent)       # Вычисляем ее приспособленность
display(bestParent)                         # Отображаем начальную строку

while True:
    # Генерируем мутированных потомков от лучшего родителя,
    # найденного на данный момент
    child = mutate(bestParent)
    childFitness = get_fitness(child)
    # Если потомок имеет более высокую оценку приспособленности,
    # чем текущий лучший родитель, он становится новым лучшим родителем
    if bestFitness >= childFitness:
        continue
    display(child)
    # Цикл продолжается до тех пор, пока потомок не совпадет с целевой строкой
    if childFitness >= len(bestParent):
        break
    bestFitness = childFitness  # Обновляем лучшую приспособленность
    bestParent = child          # Обновляем лучшего родителя
