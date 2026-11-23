import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter
from skimage.exposure import match_histograms

# ИЗМЕНЕНО: путь считается от папки, где лежит скрипт, а не от текущей папки IDLE
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Lab2_Media")
IN_DIR = os.path.join(BASE, "data", "input")
OUT_DIR = os.path.join(BASE, "data", "output")
LOG_FILE = os.path.join(BASE, "logs", "history.txt")
FILTER_NAME = "Свёртка + Цветокоррекция по шаблону"

def write_log(name, status):
    """Режим 'a' дописывает в конец и сохраняет историю."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] Файл: {name} | Фильтр: {FILTER_NAME} | Статус: {status}\n"
    try:                                            # ИЗМЕНЕНО: защита от Traceback
        with open(LOG_FILE, "a", encoding="UTF-8") as f:
            f.write(line)
    except OSError:
        print("Предупреждение: не удалось записать лог")

def convolution(img):
    """Задание 1.6: свёртка с ядром 3x3 (повышение резкости)."""
    kernel = (0, -1, 0,
              -1, 5, -1,
              0, -1, 0)
    return img.filter(ImageFilter.Kernel((3, 3), kernel, scale=1, offset=0))

def color_by_template(img, template):
    """Задание 2.6: подгонка цветов под шаблон (сопоставление гистограмм)."""
    src = np.array(img)
    ref = np.array(template)
    matched = match_histograms(src, ref, channel_axis=-1)
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8))

# НОВОЕ: отдельная функция чтения картинки
def load_image(path):
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
        img.load()
    return img

# ИЗМЕНЕНО: шаблон читается здесь же, а недоступные файлы пропускаются
def pick_template(current_name):
    """Шаблон: другой файл из input, который удаётся прочитать."""
    try:
        files = sorted(os.listdir(IN_DIR))
    except OSError:
        return None
    for fname in files:
        if fname != current_name and fname.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                return load_image(os.path.join(IN_DIR, fname))
            except OSError:      # PermissionError входит в OSError
                continue
    return None

def make_template():
    """Шаблон, заданный в коде: тёплый закатный градиент."""
    h, w = 256, 256
    t = np.linspace(0, 1, w)
    r = 255 * (0.3 + 0.7 * t)
    g = 255 * (0.2 + 0.5 * t)
    b = 255 * (0.4 - 0.3 * t)
    row = np.stack([r, g, b], axis=-1)
    arr = np.tile(row, (h, 1, 1)).astype(np.uint8)
    return Image.fromarray(arr)
def main():
    name = input("Какой файл обработать (image1.jpg или image2.png)? ").strip()
    path = os.path.join(IN_DIR, name)

    # ИЗМЕНЕНО: в try читается ТОЛЬКО выбранный пользователем файл
    try:
        img = load_image(path)
    except PermissionError:
        print(f"Ошибка: Нет прав на чтение файла {name}. Обратитесь к администратору")
        write_log(name, "Ошибка доступа")
        return
    except FileNotFoundError:
        print("Ошибка: Файл не найден")
        write_log(name, "Файл не найден")
        return
    except OSError:
        print("Ошибка: Не удалось открыть файл как изображение")
        write_log(name, "Ошибка открытия")
        return

    # НОВОЕ: шаблон берётся отдельно, его ошибки не влияют на сообщение об основном файле
    template = make_template()

    result = convolution(img)
    result = color_by_template(result, template)

    # ИЗМЕНЕНО: сохранение тоже защищено
    out_path = os.path.join(OUT_DIR, "result.jpg")
    try:
        result.save(out_path, quality=95)
    except OSError:
        print("Ошибка: Не удалось сохранить результат в папку output")
        write_log(name, "Ошибка сохранения")
        return

    write_log(name, "Успешно")
    print("Готово: результат сохранён в", out_path)

if __name__ == "__main__":
    main()
