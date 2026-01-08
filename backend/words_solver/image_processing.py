import time

import cv2
import numpy as np
from keras.utils import img_to_array

from .word_search import find_words


def manual_crop(image, x_start, y_start, x_end, y_end):
    return image[y_start:y_end, x_start:x_end]


def avg_hsv(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return np.mean(hsv, axis=(0, 1))


def is_color_in_range(color, color_range):
    if not (
        isinstance(color_range, tuple)
        and len(color_range) == 2
        and all(isinstance(r, list) and len(r) == 3 for r in color_range)
    ):
        raise ValueError("color_range must be a tuple of two lists of length 3")

    lower_bounds, upper_bounds = color_range

    return all(
        lower <= channel <= upper
        for channel, lower, upper in zip(color, lower_bounds, upper_bounds)
    )


# HSV ranges for multipliers
ORANGE_RANGE = ([0, 50, 150], [30, 255, 255])      # x2, c2
PURPLE_RANGE = ([100, 50, 150], [137, 255, 255])   # x3, c3
RED_RANGE = ([35, 80, 220], [70, 130, 250])        # red corner masking


CLASS_LABELS = {
    0: "A", 1: "B", 2: "Ch", 3: "D", 4: "E", 5: "E**", 6: "F", 7: "G", 8: "H", 9: "I",
    10: "J", 11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 16: "P", 17: "R", 18: "S", 19: "Sch",
    20: "Sh", 21: "T", 22: "Ts", 23: "U", 24: "V", 25: "Y", 26: "Ya", 27: "Yu", 28: "Z", 29: "Zh",
    30: "c2", 31: "c3", 32: "hard", 33: "soft", 34: "x2", 35: "x3",
}


# Keep payload letters in Cyrillic (represented as unicode escapes to keep source ASCII-only)
TRANSLIT_TO_RUS = {
    "A": "\u0430",     # а
    "B": "\u0431",     # б
    "Ch": "\u0447",    # ч
    "D": "\u0434",     # д
    "E": "\u0435",     # е
    "E**": "\u044d",   # э
    "F": "\u0444",     # ф
    "G": "\u0433",     # г
    "H": "\u0445",     # х
    "I": "\u0438",     # и
    "J": "\u0439",     # й
    "K": "\u043a",     # к
    "L": "\u043b",     # л
    "M": "\u043c",     # м
    "N": "\u043d",     # н
    "O": "\u043e",     # о
    "P": "\u043f",     # п
    "R": "\u0440",     # р
    "S": "\u0441",     # с
    "Sch": "\u0449",   # щ
    "Sh": "\u0448",    # ш
    "T": "\u0442",     # т
    "Ts": "\u0446",    # ц
    "U": "\u0443",     # у
    "V": "\u0432",     # в
    "Y": "\u044b",     # ы
    "Ya": "\u044f",    # я
    "Yu": "\u044e",    # ю
    "Z": "\u0437",     # з
    "Zh": "\u0436",    # ж
    "hard": "\u044a",  # ъ
    "soft": "\u044c",  # ь

    # these should never form words, but keep them consistent
    "c2": "\u04412",   # с2
    "c3": "\u04413",   # с3
    "x2": "\u04452",   # х2
    "x3": "\u04453",   # х3
}

def process_image(image_path, model, trie):
    overall_start = time.time()

    t0 = time.time()
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Failed to read the image"}
    print("[Time] Read image:", round(time.time() - t0, 3), "sec")

    # Crop (current hardcoded coordinates)
    t0 = time.time()
    x_start_crop, y_start_crop, x_end_crop, y_end_crop = 37, 445, 552, 975
    cropped_image = manual_crop(image, x_start_crop, y_start_crop, x_end_crop, y_end_crop)
    print("[Time] Crop image:", round(time.time() - t0, 3), "sec")

    # Split into 5x5
    t0 = time.time()
    grid_size = 5
    image_height, image_width, _ = cropped_image.shape
    cell_height = image_height // grid_size
    cell_width = image_width // grid_size
    print("[Time] Split to grid:", round(time.time() - t0, 3), "sec")

    cells_batch = []
    cells_mapping = []
    board = []

    detected_multipliers = {}

    for row in range(grid_size):
        row_data = []
        for col in range(grid_size):
            cell_x_start = col * cell_width
            cell_y_start = row * cell_height
            cell_x_end = (col + 1) * cell_width
            cell_y_end = (row + 1) * cell_height

            cell = cropped_image[cell_y_start:cell_y_end, cell_x_start:cell_x_end]

            shift_x = int(cell_width * 0.15)
            shift_y = int(cell_height * 0.15)
            corner_size = int(cell_width * 0.2)

            # mask bottom-right red region if detected
            bottom_right_region = cell[-corner_size - shift_y: -shift_y, -corner_size - shift_x: -shift_x]
            bottom_right_hsv = avg_hsv(bottom_right_region)
            if is_color_in_range(bottom_right_hsv, RED_RANGE):
                square_size = int(cell_width * 0.27)
                cv2.rectangle(
                    cell,
                    (cell_width - square_size, cell_height - square_size),
                    (cell_width, cell_height),
                    (255, 255, 255),
                    -1,
                )

            # detect multipliers by corner colors
            top_left_region = cell[shift_y:shift_y + corner_size, shift_x:shift_x + corner_size]
            bottom_left_region = cell[-corner_size - shift_y:-shift_y, shift_x:shift_x + corner_size]

            top_left_hsv = avg_hsv(top_left_region)
            bottom_left_hsv = avg_hsv(bottom_left_region)

            multiplier = None
            if is_color_in_range(top_left_hsv, ORANGE_RANGE):
                multiplier = "x2"
            elif is_color_in_range(top_left_hsv, PURPLE_RANGE):
                multiplier = "x3"
            elif is_color_in_range(bottom_left_hsv, ORANGE_RANGE):
                multiplier = "c2"
            elif is_color_in_range(bottom_left_hsv, PURPLE_RANGE):
                multiplier = "c3"

            if multiplier:
                detected_multipliers[(row, col)] = multiplier
                large_corner = int(cell_width * 0.58)
                if multiplier in ["x2", "x3"]:
                    pts = np.array([[0, 0], [large_corner, 0], [0, large_corner]], np.int32)
                else:
                    pts = np.array([[0, cell_height], [large_corner, cell_height], [0, cell_height - large_corner]], np.int32)
                cv2.fillPoly(cell, [pts], (255, 255, 255))

            cell_resized = cv2.resize(cell, (64, 64))
            cell_array = img_to_array(cell_resized) / 255.0
            cell_array = np.expand_dims(cell_array, axis=0)

            row_data.append((None, multiplier))
            cells_batch.append(cell_array)
            cells_mapping.append((row, col, multiplier))

        board.append(row_data)

    # single batch predict for 25 cells
    t0 = time.time()
    batch = np.vstack(cells_batch)
    predictions = model.predict(batch)

    for i, (row, col, multiplier) in enumerate(cells_mapping):
        predicted_class = int(np.argmax(predictions[i]))
        letter = CLASS_LABELS[predicted_class]
        board[row][col] = (letter, multiplier)
        print("Cell (%d, %d) - Letter: %s, Multiplier: %s" % (row, col, letter, str(multiplier)))
    print("[Time] Predict cells:", round(time.time() - t0, 3), "sec")

    # re-predict cells that had multipliers (second pass)
    t0 = time.time()
    update_cells = []
    update_mapping = []

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
            predicted_class = int(np.argmax(update_predictions[i]))
            new_letter = CLASS_LABELS[predicted_class]
            board[row][col] = (new_letter, detected_multipliers[(row, col)])
            print(
                "Updated Cell (%d, %d) - Letter: %s, Multiplier: %s"
                % (row, col, new_letter, detected_multipliers[(row, col)])
            )

    print("[Time] Update cells:", round(time.time() - t0, 3), "sec")

    # build Cyrillic board
    t0 = time.time()
    board_rus = []
    for row in board:
        row_rus = []
        for letter, multiplier in row:
            rus_letter = TRANSLIT_TO_RUS.get(letter, letter)
            row_rus.append((rus_letter, multiplier))
        board_rus.append(row_rus)
    print("[Time] Build rus board:", round(time.time() - t0, 3), "sec")

    # Find words on board using Trie
    t0 = time.time()
    found_words = find_words(board_rus, trie=trie, grid_size=grid_size)
    print("[Time] Find words:", round(time.time() - t0, 3), "sec")

    # sort results
    t0 = time.time()
    sorted_words = sorted(found_words.items(), key=lambda x: x[1], reverse=True)
    print("[Time] Sort results:", round(time.time() - t0, 3), "sec")

    print("[Time] Total:", round(time.time() - overall_start, 3), "sec")

    return {"words": [{"name": word, "score": score} for word, score in sorted_words]}
