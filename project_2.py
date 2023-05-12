import cv2
import numpy as np
import os

folder_path = "Photo/"

file_names = os.listdir(folder_path)

volumes = []
for file_name in file_names:
    # Загрузка изображения
    img = cv2.imread(f"{folder_path}{file_name}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(gray, (5,5), 0)

    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(img, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(closed.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            epsilon = 0.01*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            max_contour = approx

    if max_contour is not None:
        # Нахождение конуса, построенного по контуру
        hull = cv2.convexHull(max_contour)
        hull_area = cv2.contourArea(hull)
        _, radius = cv2.minEnclosingCircle(max_contour)
        volume = 1/3 * np.pi * (radius**2) * hull_area

        result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(result_img, [max_contour], -1, (0,255,0), 2)

        volume_cm = volume * (1 / (1000**3))
        print(f"Объем объекта на фото {file_name}: {volume_cm:.2f} куб.см")
        volumes.append(volume_cm)
        cv2.imshow('result', result_img)
        cv2.waitKey(0)
    else:
        print(f"Объект не найден на фото {file_name}")

if volumes:
    avg_volume = sum(volumes) / len(volumes)
    print(f"Средний объем объекта на всех фотографиях: {avg_volume:.2f} куб.см")
else:
    print("Объекты не найдены на фотографиях")

