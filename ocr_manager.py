import cv2
import pytesseract
import easyocr
import os

tess_cmd = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe\tesseract.exe"
)
if os.path.isfile(tess_cmd):
    pytesseract.pytesseract.tesseract_cmd = tess_cmd
# On Linux/macOS tesseract is on PATH by default — no assignment needed.

class OCRManager:

    def __init__(self):
        print("Initializing OCR engines...")

        self.easy_reader = easyocr.Reader(['en'], gpu=False)

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------
    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        return gray

    # --------------------------------------------------------
    # TESSERACT
    # --------------------------------------------------------
    def run_tesseract(self, image):
        config = "--oem 3 --psm 6"
        return pytesseract.image_to_string(image, config=config)

    # --------------------------------------------------------
    # EASYOCR
    # --------------------------------------------------------
    def run_easyocr(self, image):
        results = self.easy_reader.readtext(image)
        lines = [text for _, text, _ in results]
        return "\n".join(lines)

    # --------------------------------------------------------
    # MAIN
    # --------------------------------------------------------
    def run(self, image, image_name="output"):

        processed = self.preprocess(image)

        tesseract_text = self.run_tesseract(processed)
        easyocr_text = self.run_easyocr(processed)

        # IMPORTANT: DO NOT OVER-MERGE (this caused your GSTIN loss)
        combined = tesseract_text + "\n" + easyocr_text

        os.makedirs("ocr_output", exist_ok=True)
        file_path = f"ocr_output/{image_name}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(combined)

        return {
            "tesseract": tesseract_text,
            "easyocr": easyocr_text,
            "combined": combined,
            "file_path": file_path
        }