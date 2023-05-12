import cv2
import numpy as np
import imutils

# Загрузка изображения и изменение размера
image = cv2.imread("photo/1.jpg")
image = imutils.resize(image, width=500)

# Преобразование в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Фильтрация шумов
img_blur = cv2.GaussianBlur(gray, (5,5), 0)

# Применение алгоритма Canny для детектирования границ объектов
edges = cv2.Canny(img_blur, 30, 150)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Нахождение контуров объектов
contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Нахождение контура объекта, занимающего большую часть изображения
max_area = 0
max_contour = None
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > max_area:
        max_area = area
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        max_contour = approx

# Вычисление объема конуса, построенного по найденному контуру
if max_contour is not None:
    # Нахождение конуса, построенного по контуру
    hull = cv2.convexHull(max_contour)
    hull_area = cv2.contourArea(hull)
    _, radius = cv2.minEnclosingCircle(max_contour)
    volume = 1/3 * np.pi * (radius**2) * hull_area

    # Вывод найденного объекта на изображении
    result_img = image.copy()
    cv2.drawContours(result_img, [max_contour], -1, (0, 255, 0), 2)

    # Вывод объема объекта в кубических сантиметрах
    volume_cm = volume * (1 / (1000**3))
    print(f"Объем объекта: {volume_cm:.2f} куб.см")

    # Отображение результата
    cv2.imshow("Object", result_img)
    cv2.waitKey(0)
else:
    print("Объект не найден")