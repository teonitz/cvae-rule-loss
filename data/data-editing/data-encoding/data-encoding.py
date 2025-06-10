tile_to_idx = {
    'F': 0, 'B': 1, 'M': 2, 'P': 3, 'O': 4,
    'I': 5, 'D': 6, 'S': 7, 'W': 8, '-': 9
}
idx_to_tile = {v: k for k, v in tile_to_idx.items()}
num_tiles = len(tile_to_idx)

ROOM_HEIGHT = 16
ROOM_WIDTH = 11

def one_hot_encode(tile_char):
    one_hot_vector = np.zeros(num_tiles, dtype=np.float32)
    idx = tile_to_idx.get(tile_char)
    if idx is not None:
        one_hot_vector[idx] = 1.0
    else:
        print(f"Предупреждение: Неизвестный символ тайла '{tile_char}'. Возвращен нулевой вектор.")
    return one_hot_vector

def load_and_preprocess_room(filepath):
    room_grid_chars = []
    with open(filepath, 'r') as f:
        for line in f:
            room_grid_chars.append(list(line.strip()))

    current_height = len(room_grid_chars)
    current_width = len(room_grid_chars[0]) if current_height > 0 else 0

    if current_height != ROOM_HEIGHT or current_width != ROOM_WIDTH:
        raise ValueError(
            f"Размер комнаты в файле {filepath} ({current_height}x{current_width}) "
            f"не соответствует ожидаемому ({ROOM_HEIGHT}x{ROOM_WIDTH})."
        )

    encoded_room = np.zeros((ROOM_HEIGHT, ROOM_WIDTH, num_tiles), dtype=np.float32)
    for r in range(ROOM_HEIGHT):
        for c in range(ROOM_WIDTH):
            char = room_grid_chars[r][c]
            encoded_room[r, c] = one_hot_encode(char)

    return encoded_room

def extract_door_info(room_filepath):
    has_door_top = 0
    has_door_bottom = 0
    has_door_left = 0
    has_door_right = 0

    with open(room_filepath, 'r') as f:
        lines = [line.strip() for line in f]

    if 'D' in lines[0] or 'D' in lines[1]:
        has_door_top = 1

    if ROOM_HEIGHT >= 2 and ('D' in lines[ROOM_HEIGHT - 1] or 'D' in lines[ROOM_HEIGHT - 2]):
        has_door_bottom = 1

    for r in range(ROOM_HEIGHT):
        if ROOM_WIDTH >= 2:
            if lines[r][0] == 'D' or lines[r][1] == 'D':
                has_door_left = 1
        elif lines[r][0] == 'D':
            has_door_left = 1

        if ROOM_WIDTH >= 2:
            if lines[r][ROOM_WIDTH - 1] == 'D' or lines[r][ROOM_WIDTH - 2] == 'D':
                has_door_right = 1
        elif lines[r][ROOM_WIDTH - 1] == 'D':
            has_door_right = 1

    return np.array([has_door_top, has_door_bottom, has_door_left, has_door_right], dtype=np.float32)

def load_full_dataset(base_rooms_edited_path):
        all_encoded_rooms = []
        all_door_info = []

        for root, dirs, files in os.walk(base_rooms_edited_path):
            for file in files:
                if file.endswith('.txt'):
                    filepath = os.path.join(root, file)
                    try:
                        encoded_room = load_and_preprocess_room(filepath)
                        door_info = extract_door_info(filepath)

                        all_encoded_rooms.append(encoded_room)
                        all_door_info.append(door_info)
                    except ValueError as e:
                        print(f"Ошибка при обработке файла {filepath}: {e}")
                    except Exception as e:
                        print(f"Непредвиденная ошибка при обработке файла {filepath}: {e}")

        if all_encoded_rooms:
            X_data = np.array(all_encoded_rooms, dtype=np.float32)
            Y_data = np.array(all_door_info, dtype=np.float32)
            print(f"Загружено {len(all_encoded_rooms)} комнат.")
            print(f"Размер X_data (комнаты): {X_data.shape}")
            print(f"Размер Y_data (информация о дверях): {Y_data.shape}")
        else:
            print("Не найдено комнат для загрузки.")
            X_data = np.array([])
            Y_data = np.array([])

        return X_data, Y_data

def one_hot_decode(one_hot_vector):
    idx = np.argmax(one_hot_vector)
    return idx_to_tile.get(idx, '?')

def room_array_to_text(room_array):
    if room_array.ndim == 3 and room_array.shape[2] == num_tiles:
        height, width, _ = room_array.shape
        text_grid = []
        for r in range(height):
            row_chars = [one_hot_decode(room_array[r, c]) for c in range(width)]
            text_grid.append("".join(row_chars))
    elif room_array.ndim == 2:
        height, width = room_array.shape
        text_grid = []
        for r in range(height):
            row_chars = [idx_to_tile.get(room_array[r, c], '?') for c in range(width)]
            text_grid.append("".join(row_chars))
    else:
        raise ValueError(f"Неподдерживаемый формат room_array для room_array_to_text. Ожидается 2D или 3D массив, получен {room_array.ndim}D с формой {room_array.shape}")
    return "\n".join(text_grid)