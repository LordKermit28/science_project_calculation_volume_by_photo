import math

import cv2
import numpy as np
import os

def find_red_markers(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower_red, upper_red)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    marker_coords = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Область маркера должна быть достаточно большой, чтобы исключить шум
            M = cv2.moments(cnt)
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
            marker_coords.append((center_x, center_y))
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), -1)

    if len(marker_coords) >= 2:
        # Расчет расстояния между двумя первыми найденными маркерами
        x1, y1 = marker_coords[0]
        x2, y2 = marker_coords[1]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        cv2.putText(image, f"Distance: {distance:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.namedWindow('Маркеры', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Маркеры', 800, 600)
    cv2.imshow('Маркеры', image)

    return distance


def calculate_distance(pixel_distance, real_distance):
    pixel_per_cm = pixel_distance / real_distance
    return pixel_per_cm

image = cv2.imread('Photo/2/marked/IMG20231128134200.jpg')
marker_coords = find_red_markers(image)

def main():
    folder_path = "Photo/2/marked"
    file_names = os.listdir(folder_path)
    volumes = []
    for file_name in file_names:
        img_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(img_path):
            print(f"Файл {file_name} не существует в папке {folder_path}")
            continue

        def find_longest_distance(points):
            max_distance = 0
            for i in range(len(points)):
                p1 = points[i]
                for j in range(i + 1, len(points)):
                    p2 = points[j]
                    distance = np.linalg.norm(p1 - p2)
                    if distance > max_distance:
                        max_distance = distance
            return max_distance

        # Загружаем изображение, масштабируем и преобразуем в оттенки серого
        img = cv2.imread(img_path)
        # поиск красных маркеров
        distance_pixels = find_red_markers(img)
        distance_cm = int(input("Введите расстояние в см между маркерами: "))
        distance_coefficient = distance_pixels/distance_cm
        print(distance_pixels, distance_cm, distance_coefficient)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Выполняем обработку изображения для выделения контуров
        img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(img, 50, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Находим контуры на обработанном изображении
        contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Вычисляем объем и выводим результаты
        max_contour = None
        max_contour_distance = 0

        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            if len(approx) == 3:
                distance = find_longest_distance(approx)
                if distance > max_contour_distance:
                    max_contour_distance = distance
                    max_contour = approx

        if max_contour is not None:
            result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(result_img, [max_contour], -1, (0, 255, 0), 2)
            cv2.namedWindow('Обработанное изображение', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Обработанное изображение', 800, 600)
            cv2.imshow('Обработанное изображение', result_img)
            cv2.waitKey(0)
            print(f"Наибольшее расстояние между противоположными точками внутри контура: {max_contour_distance}")


        if max_contour is not None:
            hull = cv2.convexHull(max_contour)
            hull_area = cv2.contourArea(hull)
            _, radius = cv2.minEnclosingCircle(max_contour)
            print(f"radius: {radius}")
            volume = 1 / 3 * np.pi * (radius ** 2) * (radius*1.5)
            volume_cm = volume/distance_coefficient**3
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


