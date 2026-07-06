from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from ocr_manager import OCRManager

ocr = OCRManager()

MODEL_PATH = "Newend.pt"
model = YOLO(MODEL_PATH)

HEADER_CROP_FOLDER = "header_crops"
OCR_OUTPUT_FOLDER = "ocr_output"

os.makedirs(HEADER_CROP_FOLDER, exist_ok=True)
os.makedirs(OCR_OUTPUT_FOLDER, exist_ok=True)


def extract_header(image_path):

    image_name = Path(image_path).stem
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]

    results = model(image_path, verbose=False)

    table_boxes = []

    # --------------------------------------------------------
    # detect tables
    # --------------------------------------------------------
    for r in results:
        for box in r.boxes:

            label = model.names[int(box.cls[0])]

            if label == "Table":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                table_boxes.append((y1, y2))

    # --------------------------------------------------------
    # crop non-table region
    # --------------------------------------------------------
    if len(table_boxes) == 0:
        first_y = int(h * 0.6)
        last_y = h
        combined = img
    else:
        table_boxes.sort()
        first_y = table_boxes[0][0]
        last_y = table_boxes[-1][1]

        header = img[0:first_y, :]
        footer = img[last_y:h, :]

        combined = cv2.vconcat([header, footer]) if footer.size else header

    # --------------------------------------------------------
    # save crop
    # --------------------------------------------------------
    crop_path = f"{HEADER_CROP_FOLDER}/{image_name}_non_table.png"
    cv2.imwrite(crop_path, combined)

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------
    ocr_result = ocr.run(combined, image_name=image_name)

    text = ocr_result["combined"]

    txt_path = ocr_result["file_path"]

    return {
        "ocr_text": text,
        "ocr_file": txt_path,
        "header_crop_path": crop_path,
        "first_table_y": first_y,
        "last_table_y": last_y
    }