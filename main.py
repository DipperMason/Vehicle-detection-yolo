import torch
import tkinter as tk
from tkinter import filedialog
import os
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ========================= НАСТРОЙКИ =========================

DATA_YAML = "data.yaml"
EXPERIMENT_NAME = "vehicle_detection"

# ========================= ОБУЧЕНИЕ =========================

model = YOLO("yolov8s.pt")

train_results = model.train(
    data=DATA_YAML,
    epochs=20,
    imgsz=640,
    batch=8,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=4,
    name=EXPERIMENT_NAME,
    plots=True,
    close_mosaic=10,
    patience=10,
    save=True
)

# ========================= ЭКСПОРТ В EXCEL =========================

csv_path = os.path.join("runs", "detect", EXPERIMENT_NAME, "results.csv")
xlsx_path = os.path.join("runs", "detect", EXPERIMENT_NAME, "results.xlsx")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df.to_excel(xlsx_path, index=False)
    print(f"\nExcel-файл сохранён корректно: {xlsx_path}")
else:
    print("Файл results.csv не найден")

# ========================= ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ =========================

best_model_path = os.path.join(
    "runs", "detect", EXPERIMENT_NAME, "weights", "best.pt"
)
model = YOLO(best_model_path)

# ========================= ФУНКЦИИ =========================

def traffic_density(count):
    if count <= 5:
        return "Низкая"
    elif count <= 15:
        return "Средняя"
    else:
        return "Высокая"


def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Выберите изображение для анализа",
        filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    return file_path


def draw_text_with_background(draw, text, position, font,
                              text_color="white", bg_color="blue"):
    x, y = position
    bbox = draw.textbbox((x, y), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    padding = 8

    draw.rectangle(
        [x - padding, y - padding,
         x + width + padding, y + height + padding],
        fill=bg_color
    )
    draw.text((x, y), text, font=font, fill=text_color)


def process_image(image_path):
    if not image_path or not os.path.exists(image_path):
        print("Изображение не выбрано или не существует.")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("Не удалось загрузить изображение.")
        return

    results = model.predict(frame, conf=0.25, iou=0.5, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 32)
        font_small = ImageFont.truetype("arial.ttf", 28)
    except IOError:
        font = ImageFont.load_default()
        font_small = font

    vehicle_count = 0

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)

        if cls_id == 0:
            vehicle_count += 1
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            draw.rectangle([x1, y1, x2, y2], outline="blue", width=5)

            label_text = "vehicle"
            text_y = y1 - 40 if y1 - 40 > 10 else y1 + 10
            draw_text_with_background(
                draw, label_text, (x1 + 10, text_y), font
            )

    density = traffic_density(vehicle_count)

    stats = [
        f"Транспорт: {vehicle_count}",
        f"Плотность: {density}"
    ]

    y_offset = 15
    for line in stats:
        draw.rectangle([10, y_offset - 5, 450, y_offset + 45], fill="black")
        draw.text((20, y_offset), line, font=font_small, fill="white")
        y_offset += 55

    result_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLOv8 – Обнаружение транспорта", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" + "=" * 40)
    print("   РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("=" * 40)
    print(f"Количество транспорта: {vehicle_count}")
    print(f"Плотность трафика:     {density}")
    print("=" * 40)

    print("\nМетрики лучшей модели (валидация):")
    print(f"Precision: {train_results.results_dict['metrics/precision(B)']:.3f}")
    print(f"Recall:    {train_results.results_dict['metrics/recall(B)']:.3f}")
    print(f"mAP50:     {train_results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"mAP50-95:  {train_results.results_dict['metrics/mAP50-95(B)']:.3f}")


# ========================= ЗАПУСК =========================

if __name__ == "__main__":
    print("Выберите изображение для анализа плотности трафика...")
    img_path = select_image()
    process_image(img_path)
