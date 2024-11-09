# Импорт необходимых библиотек
from ultralytics import YOLO
import torch
import tqdm
import pandas as pd
import numpy as np
import cv2


def start_processing(path, iden):
    op = None

    
    # Функция для проверки наложения по принципу IOU
    def box_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(
            0, min(y2, y4) - max(y1, y3)
        )

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area
        return iou

    def get_bounding_rect(contour):  # функция для отрисовки определенных областей
        x, y, w, h = cv2.boundingRect(contour)
        return (x, y, x + w, y + h)

    def get_intersection(rect1, rect2):  # функция для нахождения координат пересечения областей (функция выше)
        x1, y1, x1_end, y1_end = rect1
        x2, y2, x2_end, y2_end = rect2
        x_int1 = max(x1, x2)
        y_int1 = max(y1, y2)
        x_int2 = min(x1_end, x2_end)
        y_int2 = min(y1_end, y2_end)
        if x_int1 < x_int2 and y_int1 < y_int2:
            return (x_int1, y_int1, x_int2, y_int2)
        else:
            return None

    def subtract_contours(contour1, contour2):  # функция для объединения контуров
        mask1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask1, [contour1], -1, 255, -1)
        cv2.drawContours(mask2, [contour2], -1, 255, -1)
        intersection = cv2.bitwise_and(mask1, mask2)
        mask1 = cv2.bitwise_and(mask1, cv2.bitwise_not(intersection))
        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0] if contours else []

    def filter_and_remove_overlapping_contours(contours, threshold=0.5):  # функция для нахождения пересекающихся масок
        filtered_contours = []
        for i, contour in enumerate(contours):
            keep = True
            for j, existing_contour in enumerate(filtered_contours):
                if i != j:
                    rect1 = get_bounding_rect(contour)
                    rect2 = get_bounding_rect(existing_contour)
                    intersection = get_intersection(rect1, rect2)
                    if intersection:
                        contour = subtract_contours(contour, existing_contour)
                        if len(contour) == 0:
                            keep = False
                            break
            if keep:
                filtered_contours.append(contour)
        return filtered_contours

    def remove_duplicates(masks_list):
        """Удаляет дубликаты из списка масок."""
        unique_masks = []
        masks_to_remove = set()
        for i, mask in enumerate(masks_list):
            for j, unique_mask in enumerate(unique_masks):
                if np.array_equal(mask, unique_mask):
                    masks_to_remove.add(i)
                    break
            else:
                unique_masks.append(mask)
        # Удаляем дубликаты из masks_list
        return [mask for i, mask in enumerate(masks_list) if i not in masks_to_remove]

    masks_path = []
    fin_mask = []  # Найденные маски


    torch.cuda.empty_cache()
    model = YOLO(r"Model\rectangle.pt")  # путь к модели

    img = cv2.imread(path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))  # Преобразуем кортеж в список
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(f"img{iden}.jpg", clahe_img)

    img = cv2.imread(f"img{iden}.jpg")
    blur_size = 3
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(
        mask,
        (blur_size, blur_size),
        (img.shape[1] - blur_size, img.shape[0] - blur_size),
        255,
        -1,
    )
    mask = cv2.GaussianBlur(mask, (blur_size * 2 + 1, blur_size * 2 + 1), 0)
    blurred_image = cv2.GaussianBlur(img, (blur_size * 2 + 1, blur_size * 2 + 1), 0)
    img = np.where(mask[..., np.newaxis] == 255, img, blurred_image)
    results = model(img, imgsz=2048, iou=0.001, conf=0.65, verbose=True, max_det=10000)

    if results[0].masks is not None:
        masks = results[0].masks.data
        num_masks = masks.shape[0]

        colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]

        combined_image = img.copy()

        for i in range(int(num_masks / 100 * 20)):  # Средний размер трубы на 20 % для выбора параметров модели
            color = colors[i]

        mast = 0

        total = 4
        progress = 0
        pbar = tqdm.tqdm(total=total, desc="Progress")

        results_2048 = model(img, imgsz=1024, iou=0.04, conf=0.7, verbose=False, max_det=10000)
        results_704 = model(img, imgsz=640, iou=0.04, conf=0.7, verbose=False, max_det=10000)

        boxes_2048 = results_2048[0].boxes.xyxy.cpu().numpy()
        boxes_704 = results_704[0].boxes.xyxy.cpu().numpy()

        # Объединение детекций
        combined_boxes = []
        progress += 1
        pbar.update(1)
        for box_2048 in boxes_2048:
            mask = masks[0].cpu()
            mask_resized = cv2.resize(
                np.array(mask),
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            # Наложение маски на изображение с определенной прозрачностью
            for c in range(3):  # Для каждого цветового канала
                combined_image[:, :, c] = np.where(
                    mask_resized > 0.5,
                    combined_image[:, :, c] * 1 + 255 * 0,
                    combined_image[:, :, c],
                )

            # Наложение контуров и подписей
            mask_contours_2048, _ = cv2.findContours(
                mask_resized.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            overlap = False
            for box_704 in boxes_704:
                iou = box_iou(box_2048, box_704)
                if iou > 0.5:
                    mast += 1
                    overlap = True
                    break
            if not overlap:
                combined_boxes.append(box_2048)
            else:
                combined_boxes.extend([box for box in boxes_704 if box_iou(box, box_2048) < 0.5])

        progress += 1
        pbar.update(1)

        # Перебираем объединенные боксы и создаем маски для точек
        all_contours = []

        for box in combined_boxes:
            x1, y1, x2, y2 = box.astype(int)
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            box_contour = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ])
            all_contours.append(box_contour)
        progress += 1
        pbar.update(1)


        filtered_contours = filter_and_remove_overlapping_contours(all_contours)

        # Сохранение масок
        masks_list = []
        total_area = 0
        contour_count = len(filtered_contours)

        # Подсчет общей площади
        for contour in filtered_contours:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)  # Создаем маску для текущего контура
            cv2.drawContours(mask, [contour], -1, 255, -1)  # Рисуем контур на маске
            mask_area = cv2.countNonZero(mask)  # Вычисляем площадь маски
            total_area += mask_area  # Добавляем площадь к общей

        # Вычисление средней площади
        if contour_count > 0:
            average_area = total_area / contour_count
        else:
            average_area = 0  # Избегаем деления на ноль, если контуров нет

        for contour in filtered_contours:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask_area = cv2.countNonZero(mask)
            if average_area - average_area/0.5 < mask_area < average_area/0.5:
                cv2.drawContours(combined_image, [contour], -1, (0, 255, 0), 1)
                masks_list.append(mask)

        progress += 1
        pbar.update(1)


        # Сохранение количества труб
        masks_list = remove_duplicates(masks_list)

    else:
        return 0, 0
    return combined_image, len(masks_list)

