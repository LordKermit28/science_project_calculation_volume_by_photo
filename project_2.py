import cv2
import numpy as np
import os


def main():
    folder_path = "Photo/"
    new_size = (2560, 1152)

    file_names = os.listdir(folder_path)

    volumes = []
    for file_name in file_names:
        img_path = f"{folder_path}{file_name}"
        if not os.path.isfile(img_path):
            print(f"Файл {file_name} не существует в папке {folder_path}")
            continue

        # Загружаем изображение, масштабируем и преобразуем в оттенки серого
        img = cv2.imread(img_path)
        img = cv2.resize(img, new_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Выполняем обработку изображения для выделения контуров
        img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(img, 50, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Находим контуры на обработанном изображении
        contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Вычисляем объем и выводим результаты
        max_area = 0
        max_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_contour = cnt
        if max_contour is not None:
            hull = cv2.convexHull(max_contour)
            hull_area = cv2.contourArea(hull)
            _, radius = cv2.minEnclosingCircle(max_contour)
            volume = 1 / 3 * np.pi * (radius ** 2) * hull_area
            volume_cm = volume * (1 / (1000 ** 3))
            print(f"Объем объекта на фото {file_name}: {volume_cm:.2f} куб.см")
            volumes.append(volume_cm)

            result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(result_img, [max_contour], -1, (0, 255, 0), 2)
            cv2.namedWindow('Обработанное изображение', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Обработанное изображение', 800, 600)
            cv2.imshow('Обработанное изображение', result_img)
            cv2.waitKey(0)
        else:
            print(f"Объект не найден на фото {file_name}")

    if volumes:
        avg_volume = sum(volumes) / len(volumes)
        print(f"Средний объем объекта на всех фотографиях: {avg_volume:.2f} куб.см")
    else:
        print("Объекты не найдены на фотографиях")


if __name__ == "__main__":
    main()