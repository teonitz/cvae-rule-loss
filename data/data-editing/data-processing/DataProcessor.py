# --- Constants for room processing ---
VOID_CHAR = '-'
DEFAULT_ROOM_ROWS = 16
DEFAULT_ROOM_COLS = 11

class DataProcessor:
    def __init__(self, void_char=VOID_CHAR, default_room_dims=(DEFAULT_ROOM_ROWS, DEFAULT_ROOM_COLS)):
        self.void_char = void_char
        self.default_room_rows = default_room_dims[0]
        self.default_room_cols = default_room_dims[1]

    def _read_level_text(self, filepath: str) -> list[list[str]]:
        """Reads a text level file into a 2D list of characters."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Level text file not found at {filepath}")
        with open(filepath, 'r') as f:
            lines = [list(line.strip()) for line in f if line.strip()]
        return lines

    def _find_first_room_dimensions(self, level_grid: list[list[str]]) -> tuple[int, int]:
        """
        Always returns the predefined default room dimensions (16 rows, 11 columns).
        Previous detection logic is removed/commented out.
        """
        print(f"Using fixed room dimensions: {self.default_room_rows} rows x {self.default_room_cols} cols.")
        return self.default_room_rows, self.default_room_cols

    def _pad_level_grid(self, level_grid: list[list[str]], target_rows: int, target_cols: int) -> list[list[str]]:
        """
        Pads the level grid to be perfectly divisible by target_rows and target_cols.
        """
        current_rows = len(level_grid)
        if current_rows == 0:
            return []
        current_cols = len(level_grid[0])

        padded_rows = ((current_rows + target_rows - 1) // target_rows) * target_rows
        padded_cols = ((current_cols + target_cols - 1) // target_cols) * target_cols

        padded_grid = []
        for r in range(padded_rows):
            row = []
            for c in range(padded_cols):
                if r < current_rows and c < current_cols:
                    row.append(level_grid[r][c])
                else:
                    row.append(self.void_char)
            padded_grid.append(row)
        return padded_grid

    def process_text_level_to_rooms(self, full_level_text_path: str, output_base_dir: str):
        """
        Reads a full text level and splits it into room files.
        The room dimensions for splitting are always 16x11 (rows x cols).
        Rooms are saved into a subdirectory named after the input file.
        """
        print(f"\n--- Starting processing for level: {os.path.basename(full_level_text_path)} ---")
        try:
            level_grid = self._read_level_text(full_level_text_path)
        except FileNotFoundError as e:
            print(e)
            return

        if not level_grid:
            print(f"Level grid from {os.path.basename(full_level_text_path)} is empty. Skipping.")
            return

        standard_room_rows, standard_room_cols = self.default_room_rows, self.default_room_cols
        print(f"Using fixed room dimensions: {standard_room_rows}x{standard_room_cols} for splitting.")

        padded_level_grid = self._pad_level_grid(level_grid, standard_room_rows, standard_room_cols)

        level_name = os.path.splitext(os.path.basename(full_level_text_path))[0]
        output_level_dir = os.path.join(output_base_dir, level_name)
        os.makedirs(output_level_dir, exist_ok=True)
        print(f"Saving extracted rooms to: {output_level_dir}")

        room_count = 0
        total_rows = len(padded_level_grid)
        total_cols = len(padded_level_grid[0])

        for r_grid_idx in range(total_rows // standard_room_rows):
            r_start = r_grid_idx * standard_room_rows
            for c_grid_idx in range(total_cols // standard_room_cols):
                c_start = c_grid_idx * standard_room_cols

                room_data = []
                is_room_empty = True
                for r_offset in range(standard_room_rows):
                    row_content = []
                    for c_offset in range(standard_room_cols):
                        char = padded_level_grid[r_start + r_offset][c_start + c_offset]
                        row_content.append(char)
                        if char != self.void_char:
                            is_room_empty = False
                    room_data.append("".join(row_content))

                if not is_room_empty:
                    room_file_name = f"room_{room_count:04d}_{r_grid_idx}_{c_grid_idx}.txt"
                    with open(os.path.join(output_level_dir, room_file_name), 'w') as out_f:
                        for line in room_data:
                            out_f.write(line + '\n')
                    room_count += 1
        print(f"Extracted {room_count} rooms from {os.path.basename(full_level_text_path)} using fixed {standard_room_rows}x{standard_room_cols} chunks.")