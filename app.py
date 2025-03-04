from flask import Flask, request, send_from_directory
import cv2
import numpy as np
import itertools
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Инициализация Flask-приложения
app = Flask(__name__, static_folder='static')

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

# === 1. Функция для обрезки изображения ===
def manual_crop(image, x_start, y_start, x_end, y_end):
    return image[y_start:y_end, x_start:x_end]

# === 2. Функция для усреднения HSV цвета ===
def avg_hsv(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return np.mean(hsv, axis=(0, 1))  # Усредняем по всем пикселям

# === 3. Проверяем, попадает ли цвет в нужный диапазон ===
def is_color_in_range(color, color_range):
    # Проверяем, что color_range имеет правильную структуру
    if not (
        isinstance(color_range, tuple)
        and len(color_range) == 2
        and all(isinstance(r, list) and len(r) == 3 for r in color_range)
    ):
        raise ValueError("color_range must be a tuple of two lists of length 3")

    # Распаковываем color_range
    lower_bounds, upper_bounds = color_range

    # Проверяем, что все каналы цвета попадают в диапазон
    return all(
        lower <= channel <= upper
        for channel, lower, upper in zip(color, lower_bounds, upper_bounds)
    )

# === 4. Диапазоны цветов множителей ===
orange_range = ([0, 50, 150], [30, 255, 255])  # Оранжевый (x2, c2)
purple_range = ([100, 50, 150], [137, 255, 255])  # Фиолетовый (x3, c3)


# === 5. Основная логика обработки изображения ===
def process_image(image_path):
    # === Загружаем модель нейросети ===
    model = load_model('letter_recognition_model.h5')

    # Чтение изображения
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Failed to read the image"}

    # === 7. Обрезаем изображение (идеальные координаты) ===
    x_start, y_start, x_end, y_end = 37, 445, 552, 975
    cropped_image = manual_crop(image, x_start, y_start, x_end, y_end)

    # === 8. Разделяем поле на 5x5 ===
    GRID_SIZE = 5
    image_height, image_width, _ = cropped_image.shape
    cell_height = image_height // GRID_SIZE
    cell_width = image_width // GRID_SIZE

    # === 9. Метки классов ===
    class_labels = {
        0: 'A', 1: 'B', 2: 'Ch', 3: 'D', 4: 'E', 5: 'E**', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
        10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'R', 18: 'S', 19: 'Sch',
        20: 'Sh', 21: 'T', 22: 'Ts', 23: 'U', 24: 'V', 25: 'Y', 26: 'Ya', 27: 'Yu', 28: 'Z', 29: 'Zh',
        30: 'c2', 31: 'c3', 32: 'hard', 33: 'soft', 34: 'x2', 35: 'x3'
    }

    # === 10. Распознавание букв и множителей ===
    board = []
    detected_multipliers = {}

    for row in range(GRID_SIZE):
        row_data = []
        for col in range(GRID_SIZE):
            # Вычисляем границы ячейки
            x_start = col * cell_width
            y_start = row * cell_height
            x_end = (col + 1) * cell_width
            y_end = (row + 1) * cell_height

            # Вырезаем ячейку
            cell = cropped_image[y_start:y_end, x_start:x_end]

            # === 11. Распознаем букву ===
            cell_resized = cv2.resize(cell, (64, 64))  # Изменяем размер для модели
            cell_array = img_to_array(cell_resized) / 255.0
            cell_array = np.expand_dims(cell_array, axis=0)

            prediction = model.predict(cell_array)
            predicted_class = np.argmax(prediction[0])
            letter = class_labels[predicted_class]

            # === 12. Определяем множитель ===
            shift_x = int(cell_width * 0.15)  # Смещение внутрь по X
            shift_y = int(cell_height * 0.15)  # Смещение внутрь по Y
            corner_size = int(cell_width * 0.2)  # Размер области для анализа

            # Левый верхний угол (x2, x3) - множители для букв
            top_left_region = cell[shift_y:shift_y + corner_size, shift_x:shift_x + corner_size]
            # Левый нижний угол (c2, c3) - множители для слов
            bottom_left_region = cell[-corner_size - shift_y:-shift_y, shift_x:shift_x + corner_size]

            # Получаем средний цвет в этих областях
            top_left_hsv = avg_hsv(top_left_region)
            bottom_left_hsv = avg_hsv(bottom_left_region)

            multiplier = None

            if is_color_in_range(top_left_hsv, orange_range):
                multiplier = 'x2'
            elif is_color_in_range(top_left_hsv, purple_range):
                multiplier = 'x3'
            elif is_color_in_range(bottom_left_hsv, orange_range):
                multiplier = 'c2'
            elif is_color_in_range(bottom_left_hsv, purple_range):
                multiplier = 'c3'

            # Если множитель найден - сохраняем
            if multiplier:
                detected_multipliers[(row, col)] = multiplier

                # === Рисуем треугольник, чтобы скрыть множитель ===
                corner_size = int(cell_width * 0.58)  # Размер треугольника

                if multiplier in ["x2", "x3"]:  # Верхний левый угол
                    pts = np.array([[0, 0], [corner_size, 0], [0, corner_size]], np.int32)
                else:  # Нижний левый угол (c2, c3)
                    pts = np.array([[0, cell_height], [corner_size, cell_height], [0, cell_height - corner_size]],
                                   np.int32)

                cv2.fillPoly(cell, [pts], (255, 255, 255))  # Заливаем белым цветом

            # === 13. Добавляем в массив ===
            row_data.append((letter, multiplier))
            print(f"Cell ({row}, {col}) - Letter: {letter}, Multiplier: {multiplier}")

        board.append(row_data)

    for (row, col), multiplier in detected_multipliers.items():
        # Вырезаем клетку заново после удаления множителя
        x_start = col * cell_width
        y_start = row * cell_height
        x_end = (col + 1) * cell_width
        y_end = (row + 1) * cell_height

        cell = cropped_image[y_start:y_end, x_start:x_end]  # Берём обновлённую клетку

        # === Готовим для нейросети ===
        cell_resized = cv2.resize(cell, (64, 64))  # Приводим к нужному размеру
        cell_array = img_to_array(cell_resized) / 255.0  # Нормализуем
        cell_array = np.expand_dims(cell_array, axis=0)

        # Прогоняем через модель
        prediction = model.predict(cell_array)
        predicted_class = np.argmax(prediction[0])
        new_letter = class_labels[predicted_class]  # Получаем новую букву

        # Обновляем доску
        board[row][col] = (new_letter, multiplier)  # Перезаписываем букву

        print(f"Updated Cell ({row}, {col}) - Letter: {new_letter}, Multiplier: {multiplier}")

    # === 17. Создаём русскую версию board ===
    translit_to_rus = {
        "A": "А", "B": "Б", "Ch": "Ч", "D": "Д", "E": "Е", "E**": "Э", "F": "Ф", "G": "Г", "H": "Х", "I": "И",
        "J": "Й", "K": "К", "L": "Л", "M": "М", "N": "Н", "O": "О", "P": "П", "R": "Р", "S": "С", "Sch": "Щ",
        "Sh": "Ш", "T": "Т", "Ts": "Ц", "U": "У", "V": "В", "Y": "Ы", "Ya": "Я", "Yu": "Ю", "Z": "З", "Zh": "Ж",
        "c2": "с2", "c3": "с3", "hard": "Ъ", "soft": "Ь", "x2": "х2", "x3": "х3"
    }

    # Создаём board_rus с русскими буквами
    board_rus = [[(translit_to_rus.get(cell[0], cell[0]).lower(), cell[1]) for cell in row] for row in board]

    class TrieNode:
        """ Узел Trie (каждая буква — отдельный узел) """

        def __init__(self):
            self.children = {}
            self.is_end_of_word = False

    class Trie:
        """ Префиксное дерево (Trie) для хранения словаря """

        def __init__(self):
            self.root = TrieNode()

        def insert(self, word):
            """ Вставляем слово в Trie """
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True

        def search(self, word):
            """ Проверяем, есть ли слово в Trie """
            node = self.root
            for char in word:
                if char not in node.children:
                    return False
                node = node.children[char]
            return node.is_end_of_word

        def starts_with(self, prefix):
            """ Проверяем, есть ли слова с таким префиксом """
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return False
                node = node.children[char]
            return True  # Если дошли до конца префикса, значит он существует

    # === 2️⃣ Загружаем словарь и строим Trie ===
    with open("cleaned_filtered_russian_words.json", "r", encoding="utf-8") as f:
        words = json.load(f)

    trie = Trie()
    for word in words:
        trie.insert(word)

    print(f"✅ Trie построен! Всего слов: {len(words)}")

    GRID_SIZE = len(board_rus)

    # === 4️⃣ Функция подсчета очков ===
    def calculate_word_score(word, letter_multipliers, word_multipliers):
        total_score = 0
        for i, letter in enumerate(word):
            base_score = i + 1  # Очки = позиция буквы (с 1)

            # Множители x2 и x3
            if letter_multipliers[i] == "x2":
                base_score *= 2
            elif letter_multipliers[i] == "x3":
                base_score *= 3

            total_score += base_score

        # Множители на слово (c2, c3)
        if "c2" in word_multipliers:
            total_score *= 2
        if "c3" in word_multipliers:
            total_score *= 3

        return total_score

    # === 5️⃣ DFS-поиск слов в Trie ===
    found_words = {}

    def dfs(x, y, path, visited, letter_multipliers, word_multipliers):
        word = "".join(path)

        if not trie.starts_with(word):  # Если нет префикса — остановить поиск
            return

        if len(word) > 1 and trie.search(word):  # Если слово существует
            score = calculate_word_score(word, letter_multipliers, word_multipliers)

            # Если слово уже есть, берём лучший результат
            if word in found_words:
                found_words[word] = max(found_words[word], score)
            else:
                found_words[word] = score  # Если слова не было — просто добавляем

        # Движение во все 8 направлений
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in visited:
                next_letter, multiplier = board_rus[nx][ny]
                dfs(nx, ny, path + [next_letter], visited | {(nx, ny)},
                    letter_multipliers + [multiplier],
                    word_multipliers + ([multiplier] if multiplier in ["c2", "c3"] else []))

    # === 6️⃣ Запуск DFS с каждой буквы ===
    for i, j in itertools.product(range(GRID_SIZE), repeat=2):
        letter, multiplier = board_rus[i][j]
        dfs(i, j, [letter], {(i, j)}, [multiplier], [multiplier] if multiplier in ["c2", "c3"] else [])

    # === 7️⃣ Вывод результатов ===
    sorted_words = sorted(found_words.items(), key=lambda x: x[1], reverse=True)

    print(f"🔍 Найдено {len(sorted_words)} слов:")
    for word, score in sorted_words[:30]:
        print(f"{word} — {score} очков")

    result = {"words": [{"name": word, "score": score} for word, score in sorted_words]}
    return result

# === 6. "Endpoint" для загрузки изображения ===
@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received request with files:", request.files)  # Логируем файлы
    if 'image' not in request.files:
        return json.dumps({"error": "No image part"}), 400, {'Content-Type': 'application/json; charset=utf-8'}

    file = request.files['image']
    if file.filename == '':
        return json.dumps({"error": "No selected file"}), 400, {'Content-Type': 'application/json; charset=utf-8'}

    # Сохранение файла временно
    file_path = "temp_image.jpg"
    file.save(file_path)

    try:
        # Обработка изображения
        result = process_image(file_path)

        # Создание JSON-ответа без ASCII-экранирования
        response_data = json.dumps(result, ensure_ascii=False)
        return response_data, 200, {'Content-Type': 'application/json; charset=utf-8'}

    except Exception as e:
        # Возвращаем ошибку с кодом 500
        error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
        return error_data, 500, {'Content-Type': 'application/json; charset=utf-8'}

    finally:
        # Удаляем временный файл, если он существует
        if os.path.exists(file_path):
            os.remove(file_path)

# Запуск сервера
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)