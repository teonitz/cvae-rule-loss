def horizontal_flip_array(room_array):
    return np.flip(room_array, axis=1).copy()

def vertical_flip_array(room_array):
    return np.flip(room_array, axis=0).copy()

def rotate_180_array(room_array):
    # Поворот на 180 = горизонтальное отражение + вертикальное отражение
    return np.flip(np.flip(room_array, axis=0), axis=1).copy()

def hash_room_array(room_array):
    return hashlib.sha256(room_array.tobytes()).hexdigest()

def generate_and_save_augmented_rooms(
    base_rooms_edited_path,
    output_augmented_path,
    overwrite_existing=False
):
    print(f"Запуск аугментации данных из: {base_rooms_edited_path}")
    print(f"Сохранение аугментированных данных в: {output_augmented_path}")

    if overwrite_existing and os.path.exists(output_augmented_path):
        print(f"Очистка существующей папки: {output_augmented_path}")
        shutil.rmtree(output_augmented_path)
    os.makedirs(output_augmented_path, exist_ok=True)

    processed_count = 0
    generated_count = 0
    skipped_duplicates_count = 0

    unique_room_hashes = set()

    for root, dirs, files in os.walk(base_rooms_edited_path):
        for file in files:
            if file.endswith('.txt'):
                original_filepath = os.path.join(root, file)
                try:
                    original_room_array = load_and_preprocess_room(original_filepath)
                    original_hash = hash_room_array(original_room_array)
                    unique_room_hashes.add(original_hash)
                except Exception as e:
                    print(f"Ошибка при хешировании оригинального файла {original_filepath}: {e}")

    for root, dirs, files in os.walk(base_rooms_edited_path):
        for file in files:
            if file.endswith('.txt'):
                original_filepath = os.path.join(root, file)
                try:
                    original_room_array = load_and_preprocess_room(original_filepath)
                    processed_count += 1

                    relative_path = os.path.relpath(root, base_rooms_edited_path)
                    output_subdir = os.path.join(output_augmented_path, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)

                    augmentations = {
                        "_hflip": horizontal_flip_array(original_room_array),
                        "_vflip": vertical_flip_array(original_room_array),
                        "_rot180": rotate_180_array(original_room_array)
                    }

                    for suffix, aug_array in augmentations.items():
                        aug_hash = hash_room_array(aug_array)
                        aug_text = room_array_to_text(aug_array)

                        if aug_hash not in unique_room_hashes:
                            filename = file.replace('.txt', f'{suffix}.txt')
                            filepath = os.path.join(output_subdir, filename)
                            with open(filepath, 'w') as f:
                                f.write(aug_text)
                            unique_room_hashes.add(aug_hash)
                            generated_count += 1
                        else:
                            skipped_duplicates_count += 1

                except ValueError as e:
                    print(f"Пропущена комната {original_filepath} из-за ошибки: {e}")
                except Exception as e:
                    print(f"Непредвиденная ошибка при обработке {original_filepath}: {e}")

    print(f"Завершено. Обработано {processed_count} оригинальных комнат.")
    print(f"Сгенерировано и сохранено {generated_count} УНИКАЛЬНЫХ аугментированных комнат.")
    print(f"Пропущено {skipped_duplicates_count} аугментированных комнат, так как они были дубликатами.")