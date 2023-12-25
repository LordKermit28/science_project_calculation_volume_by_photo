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




#
# def process_image(img_path):
#     if not os.path.isfile(img_path):
#         print(f"Файл {img_path} не существует")
#         return None
#
#     img = cv2.imread(img_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Выполняем обработку изображения для выделения контуров
#     img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
#     edges = cv2.Canny(img, 50, 200)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#     contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return img, edges, contours
#
#
# def calculate_volume(img, edges, contours, pixel_to_real_ratio):
#     volumes = []
#     max_perimeter = 0
#     max_contour = None
#     for cnt in contours:
#         perimeter = cv2.arcLength(cnt, True)
#         if perimeter > max_perimeter:
#             max_perimeter = perimeter
#             max_contour = cnt
#
#     if max_contour is not None:
#         hull = cv2.convexHull(max_contour)
#         hull_area = cv2.contourArea(hull)
#
#         _, radius = cv2.minEnclosingCircle(max_contour)
#         real_radius = radius * pixel_to_real_ratio
#
#         volume = 4 / 3 * np.pi * (real_radius ** 3)
#         volume_cm = volume * (1 / (1000 ** 3))
#
#         print(f"Объем объекта: {volume_cm:.2f} куб.см")
#         volumes.append(volume_cm)
#
#         result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#         cv2.drawContours(result_img, [max_contour], -1, (0, 255, 0), 2)
#
#         cv2.namedWindow('Обработанное изображение', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('Обработанное изображение', 800, 800)
#         cv2.imshow('Обработанное изображение', result_img)
#     return volumes
#
# #
# # def process_folder(folder_path, new_size):
# #     file_names = os.listdir(folder_path)
# #     volumes = []
# #     for file_name in file_names:
# #         img_path = os.path.join(folder_path, file_name)
# #         result = process_image(img_path, new_size)
# #         if result is not None:
# #             img, edges, contours = result
# #             volume_cm = calculate_volume(img, edges, contours)
# #             volumes.append(volume_cm)
# #     return volumes
#
#
# folder_path = "Photo/2/marked"
# new_size = (1800, 4000)
#
#
# def find_red_markers(image):
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     lower_red = np.array([0, 50, 50])
#     upper_red = np.array([10, 255, 255])
#     mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
#     lower_red = np.array([170, 50, 50])
#     upper_red = np.array([180, 255, 255])
#     mask2 = cv2.inRange(hsv_image, lower_red, upper_red)
#     mask = cv2.bitwise_or(mask1, mask2)
#
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     marker_coords = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 100:  # Область маркера должна быть достаточно большой, чтобы исключить шум
#             M = cv2.moments(cnt)
#             center_x = int(M['m10'] / M['m00'])
#             center_y = int(M['m01'] / M['m00'])
#             marker_coords.append((center_x, center_y))
#             cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
#             cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), -1)
#
#     cv2.namedWindow('Маркеры', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Маркеры', 800, 600)
#     cv2.imshow('Маркеры', image)
#
#     return marker_coords
#
#
# def calculate_distance(pixel_distance, real_distance):
#     pixel_per_cm = pixel_distance / real_distance
#     return pixel_per_cm
#
#
#
# image = cv2.imread('Photo/2/marked/IMG20231128134200.jpg')
# marker_coords = find_red_markers(image)
#
# if len(marker_coords) >= 2:
#     point1 = marker_coords[0]
#     point2 = marker_coords[1]
#     pixel_distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
#     real_distance = float(input("Введите расстояние в реальном мире (в см): "))
#     pixel_per_cm = calculate_distance(pixel_distance, real_distance)
#     print(f"Количество пикселей на см: {pixel_per_cm:.2f}")
# else:
#     print("На фото найдено недостаточно маркеров.")


# def run_program(image_path=folder_path):
#     image = cv2.imread(image_path)
#     marker_coords = find_red_markers(image)
#
#     if len(marker_coords) >= 2:
#         point1 = marker_coords[0]
#         point2 = marker_coords[1]
#         pixel_distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
#         real_distance = float(input("Введите расстояние в реальном мире (в см): "))
#         pixel_per_cm = calculate_distance(pixel_distance, real_distance)
#         print(f"Количество пикселей на см: {pixel_per_cm:.2f}")
#
#         processed_image = process_image(image_path)
#         if processed_image is not None:
#             img, edges, contours = processed_image
#             pixel_to_real_ratio = 1 / pixel_per_cm  # Примерный коэффициент пересчета
#
#             volume_cm = calculate_volume(img, edges, contours, pixel_to_real_ratio)
#             print(f"Объем: {volume_cm} см^3")
#
#             # Отобразить обработанное изображение
#             cv2.imshow("Processed Image", img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#     else:
#         print("На фото найдено недостаточно маркеров.")
#
#
#
# run_program()