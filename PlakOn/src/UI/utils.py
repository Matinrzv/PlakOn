# UI/utils.py

from ultralytics import YOLO
from hezar.models import Model
import cv2
from PyQt6.QtGui import QImage

_lp_detector = None
_lp_ocr = None

def load_models(
    yolo_model_path="models/lp_detector.pt",
    ocr_model_path="hezarai/crnn-fa-64x256-license-plate-recognition"
):
    """
    بارگذاری YOLO و OCR مدل‌ها (یکبار در طول اجرای برنامه)
    """
    global _lp_detector, _lp_ocr

    if _lp_detector is None:
        print("🔵 Loading YOLO model...")
        _lp_detector = YOLO(yolo_model_path)

    if _lp_ocr is None:
        print("🔵 Loading OCR model...")
        _lp_ocr = Model.load(ocr_model_path)

    return _lp_detector, _lp_ocr

def normalize_plate(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace('ي', 'ی').replace('ك', 'ک')
    text = text.replace('۰', '0').replace('۱', '1').replace('۲', '2') \
               .replace('۳', '3').replace('۴', '4').replace('۵', '5') \
               .replace('۶', '6').replace('۷', '7').replace('۸', '8') \
               .replace('۹', '9')
    return text.strip()

def format_iran_plate_simple(text: str) -> str:
    """
    خروجی OCR را به شکل ساده و بدون اسلش: '32ایران67632ب'
    """
    t = "".join(c for c in text if c.isalnum())
    if len(t) < 7:
        return "نامشخص"
    first = t[:2]
    letter = t[2]
    number = t[3:]
    return f"{first}ایران{number}{letter}"

def detect_plate_and_ocr(image_bgr):
    """
    ورودی: تصویر OpenCV (BGR)
    خروجی: (plate_text, plate_crop, bbox)
    """
    detector, ocr = load_models()
    result = detector(image_bgr)[0]

    if not result.boxes:
        return None, None, None

    box = result.boxes.data.tolist()[0]
    x1, y1, x2, y2 = map(int, box[:4])
    plate_crop = image_bgr[y1:y2, x1:x2]

    if plate_crop.size == 0:
        return None, None, None

    ocr_result = ocr.predict(plate_crop)

    plate_text = ""
    if isinstance(ocr_result, list):
        for p in ocr_result:
            if hasattr(p, "text"):
                plate_text += p.text
    elif hasattr(ocr_result, "text"):
        plate_text = ocr_result.text
    else:
        plate_text = str(ocr_result)

    plate_text = normalize_plate(plate_text)
    return plate_text, plate_crop, (x1, y1, x2, y2)




def cv_to_qimage(cv_img):
    if cv_img is None or cv_img.size == 0:
        return QImage()
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
