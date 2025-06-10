NUM_TILE_TYPES = 10 # Количество уникальных типов тайлов (должно совпадать)

char_to_int = {
    'F': 0, 'B': 1, 'M': 2, 'P': 3, 'O': 4,
    'I': 5, 'D': 6, 'S': 7, 'W': 8, '-': 9
}
int_to_char = {v: k for k, v in char_to_int.items()}

# --- 3. Загрузка обученной модели ---
print(f"Попытка загрузить модель из: {SAVED_MODEL_PATH}")
try:
    loaded_cvae_model = tf.keras.models.load_model(
        SAVED_MODEL_PATH,
        custom_objects={
            'Sampling': Sampling,
            'CVAELoss': CVAELoss
        },
        safe_mode=False
    )
    print("Модель успешно загружена.")

    if hasattr(loaded_cvae_model, 'decoder') and loaded_cvae_model.decoder is not None:
        decoder = loaded_cvae_model.decoder
        print("Декодер модели получен.")
    else:
        raise AttributeError("Не удалось получить 'decoder' из загруженной модели. Проверьте структуру CVAE класса.")


except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    print("Убедитесь, что: ")
    print("1. Файл 'best_cvae_model.keras' существует по указанному пути.")
    print("2. Классы `Sampling` и `CVAELoss` полностью скопированы и определены в этой ячейке.")
    print("3. Путь `BASE_PATH` указан верно.")

# --- 4. Функция генерации одной комнаты ---
def generate_single_room(decoder_model, latent_dim, room_condition):
    z_sample = np.random.normal(size=(1, latent_dim))

    condition_input = np.array([room_condition]).astype('float32')

    predicted_room_probabilities = decoder_model.predict([z_sample, condition_input])

    generated_room_indices = np.argmax(predicted_room_probabilities, axis=-1)

    return generated_room_indices[0]

# --- 5. Функция визуализации текстовой комнаты ---
def visualize_room_text(room_indices, int_to_char_map):
    print("\nСгенерированная комната (текст):")
    for row in room_indices:
        print("".join([int_to_char_map[idx] for idx in row]))
    print("-" * (ROOM_WIDTH + 2))

# --- 6. Функция визуализации комнаты с помощью цвета ---
def visualize_room_color(room_indices, title="Generated Room"):

    colors = [
        [0.7, 0.7, 0.7],  # Пол (светло-серый)
        [0.8, 0.8, 0.8],
        # [0.4, 0.4, 0.0],  # Блок (оливковый)
        [0.8, 0.4, 0.0],  # Враг (оранжевый)
        [0.5, 0.0, 0.5],  # Элемент (фиолетовый)
        [0.5, 0.0, 0.5],  # Элемент (фиолетовый)
        [0.5, 0.0, 0.5],  # Элемент (фиолетовый)
        [0.6, 0.3, 0.0],  # Дверь (коричневый)
        [0.8, 0.0, 0.0],  # Выход (красный)
        [0.2, 0.2, 0.2],  # Стена (темно-серый)
        [0.0, 0.0, 0.0],  # Пустота (черный)
    ]

    colored_room = np.zeros((ROOM_HEIGHT, ROOM_WIDTH, 3))
    for r in range(ROOM_HEIGHT):
        for c in range(ROOM_WIDTH):
            tile_idx = room_indices[r, c]
            colored_room[r, c] = colors[tile_idx]

    plt.figure(figsize=(6, 6))
    plt.imshow(colored_room, interpolation='nearest')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show()