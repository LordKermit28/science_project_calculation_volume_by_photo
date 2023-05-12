import os
import cv2
import numpy as np


class AdaptiveMorphology:
    def __init__(self, image_shape, opening_size=0.02, closing_size=0.02):
        self.opening_size = tuple(int(opening_size * i) for i in image_shape)
        self.closing_size = tuple(int(closing_size * i) for i in image_shape)

    def _get_kernel(self, kernel_size):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    def opening(self, image):
        kernel = self._get_kernel(self.opening_size)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def closing(self, image):
        kernel = self._get_kernel(self.closing_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


folder_path = "Photo/"
file_names = os.listdir(folder_path)
volumes = []

for file_name in file_names:
    img = cv2.imread(f"{folder_path}{file_name}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Автоматический выбор порогого значения для гистограммы яркости
    th_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Адаптивная бинаризация для выбора порогового значения для функции Canny
    edges = cv2.Canny(gray, 0, th_value * 1.5, apertureSize=3)

    # Морфологические операции для закрытия контуров
    morph = AdaptiveMorphology(gray.shape)
    closed = morph.closing(edges)

    # Нахождение контура с наибольшей площадью
    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > max_area:
            max_area = area
            max_contour = cnt

    if max_contour is not None:
        # Нахождение конуса, построенного по контуру
        hull = cv2.convexHull(max_contour)
        hull_area = cv2.contourArea(hull)
        _, radius = cv2.minEnclosingCircle(max_contour)
        volume = 1 / 3 * np.pi * (radius ** 2) * hull_area

        result_img = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(result_img, [max_contour], -1, (0, 255, 0), 2)

        volume_cm = volume * (1 / (1000 ** 3))
        print(f"Объем объекта на фото {file_name}: {volume_cm:.2f} куб.см")
        volumes.append(volume_cm)
        cv2.imshow('result', result_img)
        cv2.waitKey(0)