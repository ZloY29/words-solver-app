from flask import Flask, request, send_from_directory, jsonify
import cv2
import numpy as np
import itertools
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from dotenv import load_dotenv
import time
import threading
from werkzeug.exceptions import NotFound

# Инициализация Flask-приложения
app = Flask(__name__, static_folder='static')

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

MODEL_PATH = os.path.join(ASSETS_DIR, "letter_recognition_model.h5")
DICTIONARY_PATH = os.path.join(ASSETS_DIR, "cleaned_filtered_russian_words.json")


def get_dictionary_words():
    with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# === Загружаем модель нейросети ===
model = load_model(MODEL_PATH)

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
dictionary_words = get_dictionary_words()
trie = Trie()
for word in dictionary_words:
    trie.insert(word)

print(f"✅ Trie построен! Всего слов: {len(dictionary_words)}")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    try:
        return send_from_directory(app.static_folder, path)
    except NotFound:
        return send_from_directory(app.static_folder, "index.html")


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
##################################################################
red_range = ([35, 80, 220], [70, 130, 250])  # Красный
##################################################################


# === 5. Основная логика обработки изображения ===
def process_image(image_path):
    overall_start = time.time()  # начало работы функции

    # Чтение изображения
    t0 = time.time()
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Ошибка: не удалось прочитать изображение")
        return {"error": "Failed to read the image"}
    print(f"[Time] Чтение изображения: {time.time() - t0:.3f} сек")

    # Обрезаем изображение (идеальные координаты)
    t0 = time.time()
    x_start_crop, y_start_crop, x_end_crop, y_end_crop = 37, 445, 552, 975
    cropped_image = manual_crop(image, x_start_crop, y_start_crop, x_end_crop, y_end_crop)
    print(f"[Time] Обрезка изображения: {time.time() - t0:.3f} сек")

    # Разбиваем поле на 5x5
    t0 = time.time()
    GRID_SIZE = 5
    image_height, image_width, _ = cropped_image.shape
    cell_height = image_height // GRID_SIZE
    cell_width = image_width // GRID_SIZE
    print(f"[Time] Разбиение на клетки: {time.time() - t0:.3f} сек")

    # Метки классов
    class_labels = {
        0: 'A', 1: 'B', 2: 'Ch', 3: 'D', 4: 'E', 5: 'E**', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
        10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'R', 18: 'S', 19: 'Sch',
        20: 'Sh', 21: 'T', 22: 'Ts', 23: 'U', 24: 'V', 25: 'Y', 26: 'Ya', 27: 'Yu', 28: 'Z', 29: 'Zh',
        30: 'c2', 31: 'c3', 32: 'hard', 33: 'soft', 34: 'x2', 35: 'x3'
    }

    # Подготовка для распознавания клеток: будем собирать батч предобработанных изображений
    cells_batch = []      # для первичного распознавания всех 25 клеток
    cells_mapping = []    # для сопоставления (row, col, multiplier)
    board = []

    # Для хранения множителей, определённых на этапе распознавания клеток
    detected_multipliers = {}

    # Обходим клетки (5x5)
    for row in range(GRID_SIZE):
        row_data = []
        for col in range(GRID_SIZE):
            # Вычисляем границы ячейки
            cell_x_start = col * cell_width
            cell_y_start = row * cell_height
            cell_x_end = (col + 1) * cell_width
            cell_y_end = (row + 1) * cell_height

            # Вырезаем ячейку
            cell = cropped_image[cell_y_start:cell_y_end, cell_x_start:cell_x_end]

            # Дополнительная обработка для определения наличия множителя
            shift_x = int(cell_width * 0.15)  # Смещение внутрь по X
            shift_y = int(cell_height * 0.15)  # Смещение внутрь по Y
            corner_size = int(cell_width * 0.2)  # Размер области для анализа

            # Анализ правого нижнего угла для заливки множителя (если красный цвет обнаружен)
            bottom_right_region = cell[-corner_size - shift_y: -shift_y, -corner_size - shift_x: -shift_x]
            bottom_right_hsv = avg_hsv(bottom_right_region)
            if is_color_in_range(bottom_right_hsv, red_range):
                square_size = int(cell_width * 0.27)
                cv2.rectangle(cell, (cell_width - square_size, cell_height - square_size),
                              (cell_width, cell_height), (255, 255, 255), -1)

            # Предварительная обработка ячейки для модели
            cell_resized = cv2.resize(cell, (64, 64))
            cell_array = img_to_array(cell_resized) / 255.0
            cell_array = np.expand_dims(cell_array, axis=0)  # shape (1, 64, 64, 3)

            # Определяем множитель (анализ верхнего левого и нижнего левого углов)
            top_left_region = cell[shift_y:shift_y + corner_size, shift_x:shift_x + corner_size]
            bottom_left_region = cell[-corner_size - shift_y:-shift_y, shift_x:shift_x + corner_size]

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

            if multiplier:
                detected_multipliers[(row, col)] = multiplier
                # Если множитель найден, закрашиваем соответствующую область
                large_corner = int(cell_width * 0.58)
                if multiplier in ["x2", "x3"]:
                    pts = np.array([[0, 0], [large_corner, 0], [0, large_corner]], np.int32)
                else:
                    pts = np.array([[0, cell_height], [large_corner, cell_height], [0, cell_height - large_corner]], np.int32)
                cv2.fillPoly(cell, [pts], (255, 255, 255))

            # Добавляем placeholder в board (буква будет определена батчем)
            row_data.append((None, multiplier))
            # Сохраняем ячейку и её координаты для батчевого предсказания
            cells_batch.append(cell_array)
            cells_mapping.append((row, col, multiplier))
        board.append(row_data)

    # Объединяем все ячейки в один батч и вызываем модель один раз
    t0 = time.time()
    batch = np.vstack(cells_batch)  # shape (25, 64, 64, 3)
    predictions = model.predict(batch)
    for i, (row, col, multiplier) in enumerate(cells_mapping):
        predicted_class = np.argmax(predictions[i])
        letter = class_labels[predicted_class]
        board[row][col] = (letter, multiplier)
        print(f"Cell ({row}, {col}) - Letter: {letter}, Multiplier: {multiplier}")
    print(f"[Time] Распознавание клеток: {time.time() - t0:.3f} сек")

    # Обновление клеток с множителями в отдельном батче
    t0 = time.time()
    update_cells = []
    update_mapping = []  # (row, col)
    for (row, col), multiplier in detected_multipliers.items():
        cell_x_start = col * cell_width
        cell_y_start = row * cell_height
        cell_x_end = (col + 1) * cell_width
        cell_y_end = (row + 1) * cell_height
        cell = cropped_image[cell_y_start:cell_y_end, cell_x_start:cell_x_end]
        cell_resized = cv2.resize(cell, (64, 64))
        cell_array = img_to_array(cell_resized) / 255.0
        cell_array = np.expand_dims(cell_array, axis=0)
        update_cells.append(cell_array)
        update_mapping.append((row, col))
    if update_cells:
        update_batch = np.vstack(update_cells)
        update_predictions = model.predict(update_batch)
        for i, (row, col) in enumerate(update_mapping):
            predicted_class = np.argmax(update_predictions[i])
            new_letter = class_labels[predicted_class]
            board[row][col] = (new_letter, detected_multipliers[(row, col)])
            print(f"Updated Cell ({row}, {col}) - Letter: {new_letter}, Multiplier: {detected_multipliers[(row, col)]}")
    print(f"[Time] Обновление клеток: {time.time() - t0:.3f} сек")

    # Создаём русскую версию board
    t0 = time.time()
    translit_to_rus = {
        "A": "А", "B": "Б", "Ch": "Ч", "D": "Д", "E": "Е", "E**": "Э", "F": "Ф", "G": "Г", "H": "Х", "I": "И",
        "J": "Й", "K": "К", "L": "Л", "M": "М", "N": "Н", "O": "О", "P": "П", "R": "Р", "S": "С", "Sch": "Щ",
        "Sh": "Ш", "T": "Т", "Ts": "Ц", "U": "У", "V": "В", "Y": "Ы", "Ya": "Я", "Yu": "Ю", "Z": "З", "Zh": "Ж",
        "c2": "с2", "c3": "с3", "hard": "Ъ", "soft": "Ь", "x2": "х2", "x3": "х3"
    }
    board_rus = [[(translit_to_rus.get(cell[0], cell[0]).lower(), cell[1]) for cell in row] for row in board]
    print(f"[Time] Создание русской версии доски: {time.time() - t0:.3f} сек")

    # Функция подсчета очков (оставляем без изменений)
    def calculate_word_score(word, letter_multipliers, word_multipliers):
        total_score = 0
        for i, letter in enumerate(word):
            base_score = i + 1  # Очки = позиция буквы (с 1)
            if letter_multipliers[i] == "x2":
                base_score *= 2
            elif letter_multipliers[i] == "x3":
                base_score *= 3
            total_score += base_score
        if "c2" in word_multipliers:
            total_score *= 2
        if "c3" in word_multipliers:
            total_score *= 3
        return total_score

    # DFS-поиск слов в Trie
    t0 = time.time()
    found_words = {}

    def dfs(x, y, path, visited, letter_multipliers, word_multipliers):
        word = "".join(path)
        if not trie.starts_with(word):
            return
        if len(word) > 1 and trie.search(word):
            score = calculate_word_score(word, letter_multipliers, word_multipliers)
            if word in found_words:
                found_words[word] = max(found_words[word], score)
            else:
                found_words[word] = score
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in visited:
                next_letter, multiplier = board_rus[nx][ny]
                dfs(nx, ny, path + [next_letter], visited | {(nx, ny)},
                    letter_multipliers + [multiplier],
                    word_multipliers + ([multiplier] if multiplier in ["c2", "c3"] else []))

    for i, j in itertools.product(range(GRID_SIZE), repeat=2):
        letter, multiplier = board_rus[i][j]
        dfs(i, j, [letter], {(i, j)}, [multiplier], [multiplier] if multiplier in ["c2", "c3"] else [])
    print(f"[Time] DFS-поиск: {time.time() - t0:.3f} сек")

    # Вывод результатов
    t0 = time.time()
    sorted_words = sorted(found_words.items(), key=lambda x: x[1], reverse=True)
    print(f"🔍 Найдено {len(sorted_words)} слов:")
    for word, score in sorted_words[:30]:
        print(f"{word} — {score} очков")
    print(f"[Time] Сортировка результатов: {time.time() - t0:.3f} сек")

    overall_time = time.time() - overall_start
    print(f"⏱️ Общее время обработки изображения: {overall_time:.3f} сек")

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

    try:
        # 1) Открываем входной файл через Pillow
        img = Image.open(file.stream)  # file.stream позволяет читать данные напрямую

        # 2) Приводим к RGB (убираем альфа-канал, если он есть)
        img = img.convert("RGB")

        # 3) Изменяем размер на 590×1280 (AntiAlias = высокое качество)
        img = img.resize((590, 1280), Image.Resampling.LANCZOS)

        # 4) Сохраняем во временный файл в формате JPEG
        file_path = "temp_image.jpg"
        img.save(file_path, format="JPEG", quality=95)

    except Exception as e:
        # Если что-то пошло не так при чтении/преобразовании
        error_data = json.dumps({"error": f"Failed to preprocess image: {str(e)}"}, ensure_ascii=False)
        return error_data, 500, {'Content-Type': 'application/json; charset=utf-8'}

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
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)