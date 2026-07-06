from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "Newend.pt"
TABLE_CROP_FOLDER = "table_crops"

os.makedirs(TABLE_CROP_FOLDER, exist_ok=True)

# ============================================================
# LOAD MODELS ONCE
# ============================================================

model = YOLO(MODEL_PATH)

reader = easyocr.Reader(
    ['en'],
    gpu = False
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_iou(box1, box2):

    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    inter_x1 = max(x1a, x1b)
    inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b)
    inter_y2 = min(y2a, y2b)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area1 = (x2a - x1a) * (y2a - y1a)
    area2 = (x2b - x1b) * (y2b - y1b)

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


def remove_overlapping_tables(table_boxes, iou_threshold=0.5):

    table_boxes = sorted(
        table_boxes,
        key=lambda x: x["area"],
        reverse=True
    )

    final_tables = []

    for candidate in table_boxes:

        keep = True

        for selected in final_tables:

            iou = calculate_iou(
                candidate["bbox"],
                selected["bbox"]
            )

            if iou > iou_threshold:
                keep = False
                break

        if keep:
            final_tables.append(candidate)

    return final_tables


def extract_table(table_crop):

    results = reader.readtext(
        table_crop
    )

    words = []

    for detection in results:

        bbox, text, score = detection

        if score < 0.50:
            continue

        bbox = np.array(bbox)

        x_center = bbox[:, 0].mean()
        y_center = bbox[:, 1].mean()

        words.append(
            {
                "text": text,
                "x": round(float(x_center), 2),
                "y": round(float(y_center), 2),
                "confidence": round(float(score), 2)
            }
        )

    return words


def group_rows(words, threshold=15):

    words = sorted(words, key=lambda x: x["y"])

    rows = []
    current_row = []

    for word in words:

        if not current_row:
            current_row.append(word)
            continue

        if abs(word["y"] - current_row[0]["y"]) < threshold:
            current_row.append(word)
        else:
            rows.append(current_row)
            current_row = [word]

    if current_row:
        rows.append(current_row)

    return rows


def build_table_using_header(rows):

    if len(rows) < 2:
        return pd.DataFrame()

    header_row = rows[0]
    header_row.sort(key=lambda x: x["x"])

    header_positions = []
    header_names = []

    for item in header_row:
        header_positions.append(item["x"])
        header_names.append(item["text"])

    first_header_x = header_positions[0]

    add_left_column = False

    for row in rows[1:]:

        for item in row:

            if item["x"] < first_header_x - 50:
                add_left_column = True
                break

        if add_left_column:
            break

    if add_left_column:

        header_names = ["Column_0"] + header_names
        header_positions = [0] + header_positions

    table_rows = []

    for row in rows[1:]:

        row_dict = {
            col: ""
            for col in header_names
        }

        for item in row:

            nearest_idx = min(
                range(len(header_positions)),
                key=lambda i: abs(
                    item["x"] - header_positions[i]
                )
            )

            column_name = header_names[nearest_idx]

            if row_dict[column_name]:
                row_dict[column_name] += " " + item["text"]
            else:
                row_dict[column_name] = item["text"]

        table_rows.append(row_dict)

    return pd.DataFrame(table_rows)

# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_tables(image_path):

    image_name = Path(image_path).stem

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    results = model(
        image_path,
        verbose=False
    )

    result = results[0]

    table_boxes = []

    for box in result.boxes:

        cls_id = int(box.cls[0])

        class_name = model.names[cls_id]

        if class_name == "Table":

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            area = (x2 - x1) * (y2 - y1)

            table_boxes.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "area": area
                }
            )

    if len(table_boxes) == 0:
        return []

    final_tables = remove_overlapping_tables(
        table_boxes,
        iou_threshold=0.5
    )

    final_tables = sorted(
        final_tables,
        key=lambda x: x["bbox"][1]
    )

    all_tables = []

    for table_idx, table in enumerate(
        final_tables,
        start=1
    ):

        x1, y1, x2, y2 = table["bbox"]

        table_crop = image[
            y1:y2,
            x1:x2
        ]

        crop_path = os.path.join(
            TABLE_CROP_FOLDER,
            f"{image_name}_table_{table_idx}.png"
        )

        cv2.imwrite(
            crop_path,
            table_crop
        )

        words = extract_table(
            table_crop
        )

        rows = group_rows(
            words,
            threshold=25
        )

        df = build_table_using_header(
            rows
        )

        all_tables.append(
            {
                "table_number": table_idx,
                "crop_path": crop_path,
                "dataframe": df
            }
        )

    return all_tables

# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":

    tables = extract_tables(
        "invoice_test2.png"
    )

    print(
        f"\nTables Found: {len(tables)}"
    )

    for table in tables:

        print("\n" + "=" * 80)

        print(
            f"TABLE {table['table_number']}"
        )

        print(
            f"Crop: {table['crop_path']}"
        )

        print(
            table["dataframe"]
        )