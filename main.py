import os
import cv2
import numpy as np
from matplotlib import path as mpl_path


def get_hexagons(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Уменьшаем размер изображения для более быстрой обработки
    scaled_size = (gray.shape[1] // 2, gray.shape[0] // 2)
    scaled = cv2.resize(gray, scaled_size, interpolation=cv2.INTER_LINEAR)

    # Бинаризуем изображение
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ищем контуры на бинаризованном изображении
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hexagons = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True) * 0.05, True)
        if len(approx) == 6:
            # Проверяем, является ли контур шестиугольником
            # Если да, добавляем его координаты в список шестиугольников
            hexagons.append(approx.reshape(-1, 2) * 2)
    return hexagons


def calculate_volume(hexagons, step=10):
    # Получаем координаты точек на изображении
    points = np.concatenate(hexagons)

    # Определяем границы области на изображении
    max_x, max_y = np.max(points, axis=0)
    min_x, min_y = np.min(points, axis=0)

    # Создаем сетку для расчета объема
    x_range = np.arange(min_x, max_x, step)
    y_range = np.arange(min_y, max_y, step)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    # Заполняем зону горной породы на сетке нулями
    for hexagon in hexagons:
        path = mpl_path.Path(hexagon)
        mask = path.contains_points(np.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape)
        Z[mask] = 1

    # Рассчитываем объем
    voxel_volume = (step / 100) ** 3  # объем одного вокселя в м3
    total_volume = np.sum(Z) * voxel_volume
    return total_volume


def sort_key(filename):
    parts = filename.split(".")
    if parts[-1].lower() not in extensions:
        return float('inf')  # Use infinity as an 'unknown' extension
    digits = [part for part in parts[0] if part.isnumeric()]
    return int(''.join(digits)) if digits else float('inf')


if __name__ == "__main__":
    # Путь к папке с изображениями
    path = "photo/"

    # Получаем список файлов в папке
    extensions = {"jpg", "jpeg", "png"}
    files = os.listdir(path)
    files.sort(key=sort_key)  # использование функции sort_key в качестве ключа для сортировки списка файлов

    # Обрабатываем каждый файл
    hexagons_list = []
    for file in files:
        # Проверяем, что файл имеет расширение JPG или PNG
        if file.split(".")[-1].lower() in extensions:
            # Загружаем изображение
            image = cv2.imread(os.path.join(path, file))

            # Обрабатываем изображение
            hexagons = get_hexagons(image)
            hexagons_list.append(hexagons)

    # Рассчитываем общий объем кучи горной породы
    num_images = len(hexagons_list)

    # суммируем объемы гексагонов для всех элементов в hexagons_list
    total_volume = 0
    valid_images = 0
    for i, hexagons in enumerate(hexagons_list):
        volume = calculate_volume(hexagons)
        if valid_images == 0:
            total_volume += volume
            valid_images += 1
        elif abs(volume - total_volume / valid_images) <= 0.3 * total_volume / valid_images:
            total_volume += volume
            valid_images += 1
        print(f"Объем кучи на изображении '{files[i]}': {volume:.2f} куб.см")

    if valid_images > 0:
        avg_volume = total_volume / valid_images
        print(f"Средний объем кучи горной породы на {valid_images} изображениях: {avg_volume:.2f} куб.см")
    else:
        print("Не удалось обработать ни одно изображение с допустимым объёмом горной породы")


