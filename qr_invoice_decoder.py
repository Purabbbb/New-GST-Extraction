# qr_invoice_decoder.py

from ultralytics import YOLO
import cv2
import numpy as np
import json
import base64
import jwt
from pyzbar.pyzbar import decode
from PIL import Image, ImageFilter, ImageEnhance
from urllib.parse import urlparse, parse_qs

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH  = "Newend.pt"
CONF_THRESH = 0.25
PAD         = 15

# =====================================================
# LOAD MODEL ONCE (module-level, not per-call)
# =====================================================

print("Loading YOLO model...")
_model = YOLO(MODEL_PATH)
print(f"Model loaded. Classes: {_model.names}")

# =====================================================
# HELPERS: IMAGE PREPROCESSING FOR BETTER DECODE
# =====================================================

def _preprocess_variants(crop_bgr: np.ndarray) -> list:
    """
    Returns a list of image variants to try decoding one by one.
    Ordered from least to most aggressive processing.
    Each variant is a PIL Image.
    """
    variants = []
    h, w = crop_bgr.shape[:2]

    # Variant 1: original crop as-is
    variants.append(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))

    # Variant 2: grayscale
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(Image.fromarray(gray))

    # Variant 3: upscaled 2x (helps with small/low-res QRs)
    upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    variants.append(Image.fromarray(upscaled))

    # Variant 4: adaptive threshold (handles uneven lighting)
    thresh_adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    variants.append(Image.fromarray(thresh_adapt))

    # Variant 5: Otsu threshold + upscale
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_up = cv2.resize(thresh_otsu, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    variants.append(Image.fromarray(thresh_up))

    # Variant 6: denoised + sharpened
    denoised  = cv2.fastNlMeansDenoising(gray, h=10)
    pil_img   = Image.fromarray(denoised)
    sharpened = pil_img.filter(ImageFilter.SHARPEN)
    enhanced  = ImageEnhance.Contrast(sharpened).enhance(2.0)
    variants.append(enhanced)

    # Variant 7: morphological closing (fills gaps in QR modules)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
    variants.append(Image.fromarray(closed))

    return variants


def _try_decode_qr(crop_bgr: np.ndarray) -> str | None:
    """
    Attempts to decode a QR from a crop using multiple methods and preprocessing.
    Returns decoded string or None.
    """
    variants = _preprocess_variants(crop_bgr)

    for pil_img in variants:
        cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)

        # Method 1: pyzbar (more reliable)
        try:
            decoded_list = decode(pil_img)
            if decoded_list:
                data = decoded_list[0].data.decode("utf-8", errors="ignore").strip()
                if data:
                    return data
        except Exception:
            pass

        # Method 2: OpenCV QRCodeDetector
        try:
            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(cv_img)
            if data and data.strip():
                return data.strip()
        except Exception:
            pass

        # Method 3: OpenCV WeChatQRCode (more powerful, if available)
        try:
            wechat = cv2.wechat_qrcode_WeChatQRCode()
            texts, _ = wechat.detectAndDecode(cv_img)
            if texts and texts[0].strip():
                return texts[0].strip()
        except Exception:
            pass

    return None

# =====================================================
# HELPERS: QR DATA PARSING
# =====================================================

def _parse_qr_data(qr_data: str) -> dict | None:
    """
    Tries JWT → JSON → Base64-JSON → URL/UPI fallback.
    Returns a dict of parsed fields, or None if all methods fail.
    """

    # --- Try JWT ---
    try:
        jwt_payload = jwt.decode(qr_data, options={"verify_signature": False})
        inner = jwt_payload.get("data", jwt_payload)

        if isinstance(inner, str):
            inner = json.loads(inner)

        if isinstance(inner, dict):
            return inner
        elif isinstance(inner, (int, float, bool)):
            return {"value": inner}
        elif isinstance(inner, list):
            return {"items": json.dumps(inner)}
    except Exception:
        pass

    # --- Try JSON ---
    try:
        parsed = json.loads(qr_data)
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, list):
            return {"items": json.dumps(parsed)}
    except Exception:
        pass

    # --- Try Base64 → JSON ---
    try:
        decoded_bytes = base64.b64decode(qr_data + "==")  # padding-safe
        decoded_str   = decoded_bytes.decode("utf-8")
        parsed        = json.loads(decoded_str)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # --- Fallback: URL or UPI string (non-GST QRs) ---
    stripped = qr_data.strip()
    if stripped.startswith("http://") or stripped.startswith("https://"):
        return {"QR_Type": "URL", "URL": stripped}
    if stripped.startswith("upi://"):
        parsed_url = urlparse(stripped)
        params     = {k: v[0] for k, v in parse_qs(parsed_url.query).items()}
        return {"QR_Type": "UPI", **params}

    return None

# =====================================================
# MAIN ENTRY POINT
# =====================================================

def decode_invoice_qr(image_path: str) -> dict:
    """
    Takes a single invoice image path, detects + crops + decodes the QR,
    and returns a structured JSON-ready dict.

    Returns one of two shapes:

    Success:
        {
            "status": "success",
            "invoice_file": "invoice1.png",
            "data": { ...parsed GST/URL/UPI fields... },
            "qr_raw_data": "<raw decoded string>",
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.91
        }

    Failure:
        {
            "status": "failed",
            "invoice_file": "invoice1.png",
            "reason": "no_qr_detected" | "decode_failed" | "parse_failed" | "read_error",
            "detail": "<human readable message>"
        }
    """
    import os
    filename = os.path.basename(image_path)

    # --- Read image ---
    image = cv2.imread(image_path)
    if image is None:
        return {
            "status": "failed",
            "invoice_file": filename,
            "reason": "read_error",
            "detail": f"Image could not be read at: {image_path}"
        }

    h, w = image.shape[:2]

    # --- Run YOLO inference ---
    try:
        results = _model(image, conf=CONF_THRESH, verbose=False)
    except Exception as e:
        return {
            "status": "failed",
            "invoice_file": filename,
            "reason": "inference_error",
            "detail": str(e)
        }

    qr_found = False

    # --- Loop detections, take the first QR box ---
    for result in results:
        for box in result.boxes:
            cls        = int(box.cls[0])
            class_name = _model.names[cls]

            if class_name.upper() != "QR":
                continue

            qr_found = True
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Padding — clamped to image bounds
            x1p = max(0, x1 - PAD)
            y1p = max(0, y1 - PAD)
            x2p = min(w, x2 + PAD)
            y2p = min(h, y2 + PAD)

            qr_crop = image[y1p:y2p, x1p:x2p]

            cv2.imwrite("debug_crop.png", qr_crop)

            # --- Decode (in-memory, no file written) ---
            qr_data = _try_decode_qr(qr_crop)

            if not qr_data:
                return {
                    "status": "failed",
                    "invoice_file": filename,
                    "reason": "decode_failed",
                    "detail": "QR region detected but could not be decoded after all preprocessing attempts",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 3)
                }

            # --- Parse decoded data ---
            parsed_data = _parse_qr_data(qr_data)

            if parsed_data is None:
                return {
                    "status": "failed",
                    "invoice_file": filename,
                    "reason": "parse_failed",
                    "detail": "QR decoded but content could not be parsed into structured data",
                    "qr_raw_data": qr_data,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 3)
                }

            # --- Build clean data dict (no nested dict/list dumping issues) ---
            clean_data = {}
            for key, value in parsed_data.items():
                if isinstance(value, (dict, list)):
                    clean_data[key] = value   # keep as native JSON structure, not stringified
                else:
                    clean_data[key] = value

            return {
                "status": "success",
                "invoice_file": filename,
                "data": clean_data,
                "qr_raw_data": qr_data,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(confidence, 3)
            }

    # --- No QR class found anywhere in detections ---
    if not qr_found:
        return {
            "status": "failed",
            "invoice_file": filename,
            "reason": "no_qr_detected",
            "detail": "No QR-class bounding box found in this image"
        }









