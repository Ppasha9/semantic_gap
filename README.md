# Задача семантического разрыва
Первая реализация задачи семантического разрыва

# Постановка задачи
Пусть дана фотография, сделанная с камеры. На фотографии изображен стул на фоне дверного проема. Требуется установить, пролезает ли данный стул через дверной проем, если его нести так как сфотографировано на изображении. Если на вход передано некорректное изображение, то необходимо сообщить, какое именно из входных требований нарушено.

Предварительные требования для съемки: точка съемки должна быть примерно в пределах 3-5 метров от дверного проема.

Данные: фотографии, на которых изображен стул в различных положениях на фоне дверного проема как рядом с ним, так и в нем.

# Установка
1. Установите Anaconda с Python 3.7: https://www.anaconda.com/distribution/
   Обратите внимание на платформу (Win/Mac/Linux) и битность - обязательно 64.
2. Запустите Jupyter Notebook / Jupyter Lab сервер
   2.1 Откройте Anaconda Prompt;
   2.2 Перейдите в удобную папку;
   2.3 Запустите команду jupyter lab, либо jupyter notebook.
3. Скачайте папку с реализацией задачи и найдите её в файловом менеджере Jupyter;
4. Откройте main.ipynb;

# Что реализовано
На данный момент алгоритм делит предоставленный датасет (директория DATA_PHOTOS) на две части - тренировочный и тестовый.
Дальше для каждой фотографии из тренировочного датасета происходит поиск двери:
1. Применяется бинаризация (алгоритмы Otsu, Minimum, Mean, Yen, Li, Triangle, Isodata)
2. Используется алгоритм Canny (sigma = 2.5)
3. Применяется алгоритм Хафа для нахождения прямых линий
Результаты сохраняются в директорию COLLECTED_DATA/FILTERS, для каждого изображения создается своя поддиректория.

Это первый этап нахождения двери, в котором хочется определить какой из алгоритмов бинаризации подходит лучше всего.
На данный момент могу сказать, что алгоритм Mean показывает лучшие результаты (большее количество верного нахождения двери)

# Планы на будущее
По полученным данным из алгоритма Хафа выделить "прямоугольник" дверного проема и найти его ширину.
Найти стул на фотографии.
Определить, проходит ли стул в дверной проем.
