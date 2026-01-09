import itertools


def calculate_word_score(word, letter_multipliers, word_multipliers):
    total_score = 0
    for i, _letter in enumerate(word):
        base_score = i + 1
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


def find_words(board_rus, trie, grid_size=5):
    found_words = {}

    directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    def dfs(x, y, path, visited, letter_multipliers, word_multipliers):
        word = "".join(path)

        if not trie.starts_with(word):
            return

        if len(word) > 1 and trie.search(word):
            score = calculate_word_score(word, letter_multipliers, word_multipliers)
            prev = found_words.get(word)
            if prev is None or score > prev:
                found_words[word] = score

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in visited:
                next_letter, multiplier = board_rus[nx][ny]
                dfs(
                    nx,
                    ny,
                    path + [next_letter],
                    visited | {(nx, ny)},
                    letter_multipliers + [multiplier],
                    word_multipliers + ([multiplier] if multiplier in ["c2", "c3"] else []),
                )

    for i, j in itertools.product(range(grid_size), repeat=2):
        letter, multiplier = board_rus[i][j]
        dfs(
            i,
            j,
            [letter],
            {(i, j)},
            [multiplier],
            [multiplier] if multiplier in ["c2", "c3"] else [],
        )

    return found_words
