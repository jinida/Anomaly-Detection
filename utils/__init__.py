from pathlib import Path
from typing import List

def create_folders(base_path: str, category_name: str):
    category_dir = Path(base_path) / category_name
    last_num = 0

    if category_dir.exists():
        existing_folders = [int(d.name) for d in category_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if existing_folders:
            last_num = max(existing_folders)
    new_num = last_num + 1
    folder_name = f"{new_num:03d}"
    target_dir = category_dir / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return str(target_dir)
    