import os
import sys
from pathlib import Path

def get_dir_size(path: str) -> int:
    """Вычисляет общий размер директории итеративно (без рекурсии)."""
    total_size = 0
    stack = [path]
    while stack:
        current_path = stack.pop()
        try:
            with os.scandir(current_path) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            total_size += entry.stat().st_size
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                    except (PermissionError, OSError):
                        continue
        except (PermissionError, OSError):
            continue
    return total_size

def format_size(bytes_size: int) -> str:
    """Конвертирует байты в читаемый формат (ГБ, МБ)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024

def analyze_disk(start_path: str, limit: int = 20):
    print(f"--- Анализ директории: {start_path} ---")
    results = []
    grand_total = 0
    
    try:
        with os.scandir(start_path) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    print(f"Сканирую: {entry.name}...")
                    size = get_dir_size(entry.path)
                    results.append((entry.path, size))
                    grand_total += size
                else:
                    size = entry.stat().st_size
                    results.append((entry.path, size))
                    grand_total += size
    except (PermissionError, KeyboardInterrupt):
        print("Ошибка доступа к корневой папке. Попробуйте запустить от имени администратора.")
        return

    # Сортировка по размеру (от большего к меньшему)
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nТОП-{} САМЫХ ТЯЖЕЛЫХ ОБЪЕКТОВ:".format(limit))
    for path, size in results[:limit]:
        print(f"{format_size(size):>10} | {path}")
    
    print(f"\nОбщий размер просканированных данных: {format_size(grand_total)}")

if __name__ == "__main__":
    # Если путь передан в аргументах, используем его, иначе сканируем текущую папку
    target_path = '.'
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    
    if not os.path.exists(target_path):
        print(f"Ошибка: Путь '{target_path}' не найден.")
    else:
        analyze_disk(target_path)
