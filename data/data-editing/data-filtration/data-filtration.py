unique_room_hashes = set()
unique_room_contents = []
unique_file_count = 0

print("Начинаем сканирование и фильтрацию уникальных комнат...")

folders_to_process = deque([input_rooms_dir])

while folders_to_process:
    current_folder = folders_to_process.popleft()

    if not os.path.exists(current_folder):
        print(f"Предупреждение: Папка не существует: {current_folder}. Пропускаем.")
        continue

    try:
        for item_name in os.listdir(current_folder):
            item_path = os.path.join(current_folder, item_name)

            if os.path.isdir(item_path):
                folders_to_process.append(item_path)
            elif os.path.isfile(item_path) and item_name.endswith('.txt'):
                try:
                    with open(item_path, 'r', encoding='utf-8') as f:
                        room_content = f.read()

                    content_hash = hashlib.md5(room_content.encode('utf-8')).hexdigest()

                    if content_hash not in unique_room_hashes:
                        unique_room_hashes.add(content_hash)
                        unique_room_contents.append(room_content)
                        unique_file_count += 1
                        output_file_name = f"unique_room_{unique_file_count:05d}.txt"
                        output_file_path = os.path.join(output_unique_rooms_dir, output_file_name)
                        with open(output_file_path, 'w', encoding='utf-8') as outfile:
                            outfile.write(room_content)

                except Exception as e:
                    print(f"Ошибка при обработке файла {item_path}: {e}")
    except Exception as e:
        print(f"Ошибка при доступе к папке {current_folder}: {e}")

print(f"\nФильтрация завершена. Найдено и сохранено {unique_file_count} уникальных комнат.")
print(f"Уникальные комнаты сохранены в: {output_unique_rooms_dir}")