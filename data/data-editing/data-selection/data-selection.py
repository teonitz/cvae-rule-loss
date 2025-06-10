desired_fixed_samples = 100

print(f"\nDesired fixed number of samples: {desired_fixed_samples}")

if not os.path.exists(unique_rooms_dir):
    print(f"Error: Directory '{unique_rooms_dir}' not found. Please ensure you have run the unique room filtering script first.")
else:
    if os.path.exists(selected_rooms_dir):
        print(f"Clearing existing directory: {selected_rooms_dir}")
        shutil.rmtree(selected_rooms_dir)
    os.makedirs(selected_rooms_dir, exist_ok=True)

    unique_room_files = [f for f in os.listdir(unique_rooms_dir) if f.endswith('.txt')]
    random.shuffle(unique_room_files)

    total_unique_rooms = len(unique_room_files)
    print(f"Total unique rooms found: {total_unique_rooms}")

    num_samples_to_use = min(desired_fixed_samples, total_unique_rooms)
    if num_samples_to_use == 0 and total_unique_rooms > 0:
        print("Warning: Number of samples to use is 0. Perhaps the desired fixed samples is set to 0.")

    print(f"Will select and save {num_samples_to_use} rooms to '{selected_rooms_dir}'.")

    selected_count = 0
    for i in range(num_samples_to_use):
        source_file_name = unique_room_files[i]
        source_file_path = os.path.join(unique_rooms_dir, source_file_name)
        destination_file_path = os.path.join(selected_rooms_dir, source_file_name)

        try:
            shutil.copyfile(source_file_path, destination_file_path)
            selected_count += 1
        except Exception as e:
            print(f"Error copying file {source_file_path} to {destination_file_path}: {e}")

    print(f"\nSelection and copying completed. {selected_count} rooms saved to '{selected_rooms_dir}'.")