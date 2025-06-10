DUNGEON_ROWS = 6
DUNGEON_COLS = 6
NUM_ROOMS_TO_GENERATE_PER_CONFIG = 50

MAX_DUNGEON_ROOMS = 13
ROOM_CONNECTION_PROBABILITY = 0.5
MIN_DUNGEON_ROOMS = 8

TILE_COLORS = {
    'F': '#A0A0A0',  # Пол (светло-серый)
    'B': '#505050',  # Блок/Стена (темно-серый)
    'M': '#FFA500',  # Монстр (оранжевый)
    'P': '#FF0000',  # Начало игрока (красный)
    'O': '#00FF00',  # Препятствие (зеленый)
    'I': '#0000FF',  # Предмет (синий)
    'D': '#8B4513',  # Дверь (коричневый)
    'S': '#FFFF00',  # Секрет/Особое (желтый)
    'W': '#C0C0C0',  # Пустота/Внешняя стена (светло-серый)
    '-': '#000000'   # Пустой/Заполнитель (черный)
}

unique_tiles = list(tile_to_idx.keys())
cmap_list = [TILE_COLORS.get(tile, '#FFFFFF') for tile in unique_tiles]
cmap = mcolors.ListedColormap(cmap_list)
norm = mcolors.BoundaryNorm(np.arange(-0.5, len(unique_tiles)), cmap.N)

def hash_room_array(room_array):
    return hashlib.sha256(room_array.tobytes()).hexdigest()

# --- 1. Генерация пула комнат для каждой конфигурации дверей ---

def generate_room_pool(cvae_model_decoder, latent_dim_size, num_samples_per_config=50, save_to_dir=None):
    if cvae_model_decoder is None:
        print("Ошибка: Модель декодера CVAE не загружена. Невозможно сгенерировать пул комнат.")
        return {}

    print(f"\n--- Генерация пула комнат ({num_samples_per_config} образцов на конфигурацию дверей) ---")
    room_pool = {}
    total_generated_attempts = 0
    total_unique_rooms_saved = 0

    if save_to_dir:
        os.makedirs(save_to_dir, exist_ok=True)
        for f in os.listdir(save_to_dir):
            if f.endswith('.txt'):
                os.remove(os.path.join(save_to_dir, f))

    for i in range(2**num_door_features):
        door_config = tuple(int(b) for b in bin(i)[2:].zfill(num_door_features))
        door_info_tensor = np.array(door_config, dtype=np.float32)[np.newaxis, :]

        generated_rooms_for_config = []
        generated_hashes_for_config = set()

        attempt_count = 0
        max_attempts = num_samples_per_config * 10

        while len(generated_rooms_for_config) < num_samples_per_config and attempt_count < max_attempts:
            z_sample = np.random.normal(size=(1, latent_dim_size)).astype(np.float32)

            try:
                room_logits = cvae_model_decoder.predict([z_sample, door_info_tensor], verbose=0)
            except Exception as e:
                print(f"Ошибка при предсказании декодера для {door_config}: {e}")
                attempt_count += 1
                total_generated_attempts += 1
                continue

            generated_room_indices = np.argmax(room_logits[0], axis=-1)

            one_hot_generated_room = np.zeros((ROOM_HEIGHT, ROOM_WIDTH, num_tiles), dtype=np.float32)
            for r_idx in range(ROOM_HEIGHT):
                for c_idx in range(ROOM_WIDTH):
                    one_hot_generated_room[r_idx, c_idx, generated_room_indices[r_idx, c_idx]] = 1.0

            current_room_hash = hash_room_array(one_hot_generated_room)

            if current_room_hash not in generated_hashes_for_config:
                generated_rooms_for_config.append(one_hot_generated_room)
                generated_hashes_for_config.add(current_room_hash)
                total_unique_rooms_saved += 1
                if save_to_dir:
                    filename = f"room_config_{'_'.join(map(str, door_config))}_{len(generated_rooms_for_config)-1}.txt"
                    filepath = os.path.join(save_to_dir, filename)
                    with open(filepath, 'w') as f:
                        f.write(room_array_to_text(one_hot_generated_room))
            attempt_count += 1
            total_generated_attempts += 1

        room_pool[door_config] = generated_rooms_for_config
        print(f"  Конфигурация {door_config}: Сгенерировано {len(generated_rooms_for_config)} уникальных комнат (попыток: {attempt_count})")

    print(f"Всего уникальных комнат сгенерировано и добавлено в пул: {total_unique_rooms_saved}")
    print(f"Всего попыток генерации: {total_generated_attempts}")
    return room_pool

# --- 2. Создание абстрактной карты подземелья (на основе графа в сетке) ---

class DungeonMap:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[{'visited': False, 'door_info': (0,0,0,0), 'room_array': None}
                      for _ in range(cols)] for _ in range(rows)]
        self.num_connected_rooms = 0
        self.start_pos = None
        self.full_dungeon_pixel_map = None

    def _is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def generate_layout(self, min_rooms, max_rooms, connection_probability):
        attempts = 0
        max_layout_attempts = 1000 #

        while (self.num_connected_rooms < min_rooms or self.num_connected_rooms > max_rooms) and attempts < max_layout_attempts:
            self.grid = [[{'visited': False, 'door_info': (0,0,0,0), 'room_array': None}
                          for _ in range(self.cols)] for _ in range(self.rows)]
            self.num_connected_rooms = 0
            self.start_pos = None

            start_r, start_c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            self.start_pos = (start_r, start_c)

            q_bfs = deque([(start_r, start_c)])
            self.grid[start_r][start_c]['visited'] = True
            self.num_connected_rooms = 1

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            door_map = {
                (0, 1): (3, 2),  # current (right), neighbor (left)
                (0, -1): (2, 3), # current (left), neighbor (right)
                (1, 0): (1, 0),  # current (bottom), neighbor (top)
                (-1, 0): (0, 1)  # current (top), neighbor (bottom)
            }

            while q_bfs:
                curr_r, curr_c = q_bfs.popleft()
                random.shuffle(directions)

                for dr, dc in directions:
                    if self.num_connected_rooms >= max_rooms:
                        q_bfs.clear()
                        break

                    next_r, next_c = curr_r + dr, curr_c + dc

                    if self._is_valid(next_r, next_c) and not self.grid[next_r][next_c]['visited']:
                        if random.random() < connection_probability:
                            self.grid[next_r][next_c]['visited'] = True
                            self.num_connected_rooms += 1
                            q_bfs.append((next_r, next_c))

                            if self.num_connected_rooms >= max_rooms:
                                q_bfs.clear()
                                break

                            curr_door_idx, next_door_idx = door_map[(dr, dc)]

                            curr_door_list = list(self.grid[curr_r][curr_c]['door_info'])
                            curr_door_list[curr_door_idx] = 1
                            self.grid[curr_r][curr_c]['door_info'] = tuple(curr_door_list)

                            next_door_list = list(self.grid[next_r][next_c]['door_info'])
                            next_door_list[next_door_idx] = 1
                            self.grid[next_r][next_c]['door_info'] = tuple(next_door_list)

            self.num_connected_rooms = sum(1 for row in self.grid for cell in row if cell['visited'])
            attempts += 1

        if attempts >= max_layout_attempts:
            print(f"Предупреждение: Не удалось сгенерировать макет подземелья в диапазоне [{min_rooms}, {max_rooms}] комнат после {max_layout_attempts} попыток. Получено {self.num_connected_rooms} комнат.")
            if self.num_connected_rooms == 0:
                start_r, start_c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
                self.start_pos = (start_r, start_c)
                self.grid[start_r][start_c]['visited'] = True
                self.num_connected_rooms = 1

        if self.num_connected_rooms > max_rooms:
            print(f"Внимание: Сгенерировано {self.num_connected_rooms} комнат, что превышает максимум {max_rooms}. Обрезаем.")

        print(f"\n--- Сгенерирован абстрактный макет подземелья: {self.num_connected_rooms} комнат ---")
        print(f"Стартовая позиция: {self.start_pos}")

    def assign_rooms(self, room_pool):
        print("\n--- Присвоение комнат абстрактной карте ---")
        assigned_count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c]['visited']:
                    required_door_config = self.grid[r][c]['door_info']
                    if required_door_config in room_pool and len(room_pool[required_door_config]) > 0:
                        chosen_room = random.choice(room_pool[required_door_config])
                        self.grid[r][c]['room_array'] = chosen_room
                        assigned_count += 1
                    else:
                        print(f"Предупреждение: Нет комнат в пуле для конфигурации дверей {required_door_config} в ({r},{c}). Комната будет заполнена стенами.")
                        default_tile_idx = tile_to_idx.get('B', 0)
                        empty_room_array = np.zeros((ROOM_HEIGHT, ROOM_WIDTH, num_tiles), dtype=np.float32)
                        for r_idx in range(ROOM_HEIGHT):
                            for c_idx in range(ROOM_WIDTH):
                                empty_room_array[r_idx, c_idx, default_tile_idx] = 1.0
                        self.grid[r][c]['room_array'] = empty_room_array

        print(f"Назначено {assigned_count} реальных комнат.")

    def get_full_dungeon_pixel_map(self):
        full_dungeon_pixel_map = np.zeros(
            (self.rows * ROOM_HEIGHT, self.cols * ROOM_WIDTH), dtype=int
        )

        default_empty_tile_idx = tile_to_idx.get('-', 9)

        for r_map in range(self.rows):
            for c_map in range(self.cols):
                room_data = self.grid[r_map][c_map]['room_array']
                if room_data is not None:
                    room_indices = np.argmax(room_data, axis=-1)
                    full_dungeon_pixel_map[
                        r_map * ROOM_HEIGHT : (r_map + 1) * ROOM_HEIGHT,
                        c_map * ROOM_WIDTH : (c_map + 1) * ROOM_WIDTH
                    ] = room_indices
                else:
                    full_dungeon_pixel_map[
                        r_map * ROOM_HEIGHT : (r_map + 1) * ROOM_HEIGHT,
                        c_map * ROOM_WIDTH : (c_map + 1) * ROOM_WIDTH
                    ] = default_empty_tile_idx
        self.full_dungeon_pixel_map = full_dungeon_pixel_map
        return full_dungeon_pixel_map


    def visualize_abstract_map(self):
        fig, ax = plt.subplots(figsize=(self.cols * 1.5, self.rows * 1.5))
        ax.set_xticks(np.arange(self.cols + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.rows + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title("Абстрактная карта подземелья")

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c]['visited']:
                    color = 'lightgray'
                    if (r,c) == self.start_pos:
                        color = 'red'
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=color, zorder=0))

                    door_info = self.grid[r][c]['door_info']
                    if door_info[0] == 1:
                        ax.plot([c - 0.2, c + 0.2], [r - 0.5, r - 0.5], 'k-', lw=4, solid_capstyle='butt') # Top door
                    if door_info[1] == 1:
                        ax.plot([c - 0.2, c + 0.2], [r + 0.5, r + 0.5], 'k-', lw=4, solid_capstyle='butt') # Bottom door
                    if door_info[2] == 1:
                        ax.plot([c - 0.5, c - 0.5], [r - 0.2, r + 0.2], 'k-', lw=4, solid_capstyle='butt') # Left door
                    if door_info[3] == 1:
                        ax.plot([c + 0.5, c + 0.5], [r - 0.2, r + 0.2], 'k-', lw=4, solid_capstyle='butt') # Right door
                else:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='darkgray', zorder=0))

        ax.invert_yaxis()
        plt.show()

    def visualize_full_dungeon(self):
        print("\n--- Визуализация полного подземелья ---")
        if self.full_dungeon_pixel_map is None:
            self.get_full_dungeon_pixel_map()

        fig, ax = plt.subplots(figsize=(self.cols * ROOM_WIDTH / 10, self.rows * ROOM_HEIGHT / 10))
        ax.imshow(self.full_dungeon_pixel_map, cmap=cmap, norm=norm, origin='upper', interpolation='nearest')

        ax.set_xticks(np.arange(0, self.cols * ROOM_WIDTH, ROOM_WIDTH) - 0.5, minor=True)
        ax.set_yticks(np.arange(0, self.rows * ROOM_HEIGHT, ROOM_HEIGHT) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Сгенерированное подземелье")
        plt.show()

def generate_full_dungeon(
    cvae_model_path,
    rooms_pool_save_dir='generated_room_pool',
    dungeon_output_dir='generated_dungeons',
    dungeon_rows=DUNGEON_ROWS,
    dungeon_cols=DUNGEON_COLS,
    num_rooms_to_generate_per_config=NUM_ROOMS_TO_GENERATE_PER_CONFIG,
    min_dungeon_rooms=MIN_DUNGEON_ROOMS,
    max_dungeon_rooms=MAX_DUNGEON_ROOMS,
    room_connection_probability=ROOM_CONNECTION_PROBABILITY
):
    os.makedirs(dungeon_output_dir, exist_ok=True)

    # 2. Генерируем пул комнат
    room_pool = generate_room_pool(decoder,
                                   LATENT_DIM,
                                   num_samples_per_config=num_rooms_to_generate_per_config,
                                   save_to_dir=rooms_pool_save_dir)

    if not any(room_pool.values()):
        print("Пул комнат пуст. Невозможно сгенерировать подземелье.")
        return

    # 3. Генерируем абстрактный макет подземелья
    dungeon = DungeonMap(dungeon_rows, dungeon_cols)
    dungeon.generate_layout(min_dungeon_rooms, max_dungeon_rooms, room_connection_probability)
    dungeon.visualize_abstract_map()

    # 4. Присваиваем комнаты макету
    dungeon.assign_rooms(room_pool)

    # # 5. Визуализируем полное подземелье
    # dungeon.visualize_full_dungeon()

    # 5. Визуализируем полное подземелье
    full_dungeon_pixel_map = dungeon.get_full_dungeon_pixel_map()
    dungeon.visualize_full_dungeon()

    # 6. Сохранение текстового представления всего уровня
    print("\n--- Сохранение текстового представления всего подземелья ---")
    full_dungeon_text = room_array_to_text(full_dungeon_pixel_map)
    with open(os.path.join(dungeon_output_dir, 'full_dungeon_map.txt'), 'w') as f:
        f.write(full_dungeon_text)
    print(f"Полное текстовое представление подземелья сохранено в {os.path.join(dungeon_output_dir, 'full_dungeon_map.txt')}")

    print("\n--- Сохранение текстовых представлений отдельных комнат подземелья ---")
    for r in range(dungeon.rows):
        for c in range(dungeon.cols):
            if dungeon.grid[r][c]['visited'] and dungeon.grid[r][c]['room_array'] is not None:
                room_text = room_array_to_text(dungeon.grid[r][c]['room_array'])
                with open(os.path.join(dungeon_output_dir, f'dungeon_room_{r}_{c}.txt'), 'w') as f:
                    f.write(f"--- Комната ({r},{c}) --- Двери: {dungeon.grid[r][c]['door_info']}\n")
                    f.write(room_text + "\n\n")
            elif dungeon.grid[r][c]['room_array'] is not None:
                room_text = room_array_to_text(dungeon.grid[r][c]['room_array'])
                with open(os.path.join(dungeon_output_dir, f'dungeon_room_{r}_{c}_empty.txt'), 'w') as f:
                    f.write(f"--- Пустая ячейка ({r},{c}) --- Двери: {dungeon.grid[r][c]['door_info']}\n")
                    f.write(room_text + "\n\n")

    print(f"Процесс генерации уровня завершен. Все файлы сохранены в {dungeon_output_dir}")