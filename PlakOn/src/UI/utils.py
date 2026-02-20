# UI/utils.py

from ultralytics import YOLO
from hezar.models import Model
from hezar.preprocessors import ImageProcessor, ImageProcessorConfig
from hezar.preprocessors.preprocessor import PreprocessorsContainer
import cv2
import re
from pathlib import Path
from PyQt6.QtGui import QImage

_lp_detector = None
_lp_ocr = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "lp_detector.pt"
DEFAULT_OCR_MODEL_PATH = PROJECT_ROOT / "models" / "crnn-fa-64x256-license-plate-recognition"


class _FallbackPreprocessor:
    def __init__(self):
        # CRNN model expects grayscale images resized to 256x64.
        cfg = ImageProcessorConfig(
            gray_scale=True,
            size=(256, 64),
            rescale=1 / 255.0,
        )
        container = PreprocessorsContainer()
        container["image_processor"] = ImageProcessor(cfg)
        self.container = container


def load_models(
    yolo_model_path=DEFAULT_YOLO_MODEL_PATH,
    ocr_model_path=DEFAULT_OCR_MODEL_PATH
):
    global _lp_detector, _lp_ocr

    if _lp_detector is None:
        print("🔵 Loading YOLO model...")
        _lp_detector = YOLO(str(yolo_model_path))

    if _lp_ocr is None:
        print("🔵 Loading OCR model...")
        _lp_ocr = Model.load(str(ocr_model_path))
        image_processor = getattr(getattr(_lp_ocr, "preprocessor", None), "image_processor", None)
        if not callable(image_processor):
            _lp_ocr.preprocessor = _FallbackPreprocessor().container

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
    t = normalize_plate(text)
    if not t:
        return "نامشخص"

    # حذف نویزهای رایج OCR
    t = t.replace("ایران", "").replace("IRAN", "").replace("Iran", "")
    t = t.replace("الف", "ا")
    t = re.sub(r"[\s\-_./|]+", "", t)

    # فقط اعداد و حروف فارسی/لاتین را نگه می‌داریم
    cleaned = "".join(ch for ch in t if ch.isdigit() or ("\u0600" <= ch <= "\u06FF") or ch.isalpha())
    if not cleaned:
        return "نامشخص"

    # بهترین حالت: یک حرف فارسی وسط و قبل/بعد آن ارقام کافی
    def _is_persian_letter(ch: str) -> bool:
        return ("\u0600" <= ch <= "\u06FF") and ch.isalpha() and not ch.isdigit()

    letter_candidates = [i for i, ch in enumerate(cleaned) if _is_persian_letter(ch)]
    if not letter_candidates:
        letter_candidates = [i for i, ch in enumerate(cleaned) if ch.isalpha() and not ch.isdigit()]

    letter_idx = -1
    for idx in letter_candidates:
        before_digits = "".join(ch for ch in cleaned[:idx] if ch.isdigit())
        after_digits = "".join(ch for ch in cleaned[idx + 1:] if ch.isdigit())
        if len(before_digits) >= 2 and len(after_digits) >= 5:
            letter_idx = idx
            break
    if letter_idx == -1 and letter_candidates:
        letter_idx = letter_candidates[0]

    if letter_idx != -1:
        before_digits = "".join(ch for ch in cleaned[:letter_idx] if ch.isdigit())
        after_digits = "".join(ch for ch in cleaned[letter_idx + 1:] if ch.isdigit())
        letter = cleaned[letter_idx]

        # الگوی متداول: 2 رقم + حرف + 3 رقم + 2 رقم
        if len(before_digits) >= 2 and len(after_digits) >= 5:
            left2 = before_digits[-2:]
            middle3 = after_digits[:3]
            right2 = after_digits[3:5]
            return f"{left2} {letter} {middle3} ایران {right2}"

        # حالت جایگزین OCR: 5 رقم + حرف + 2 رقم
        if len(before_digits) >= 5 and len(after_digits) >= 2:
            left2 = before_digits[:2]
            middle3 = before_digits[2:5]
            right2 = after_digits[:2]
            return f"{left2} {letter} {middle3} ایران {right2}"

    # fallback: با فرض 7 رقم و یک حرف
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    letters = "".join(ch for ch in cleaned if ch.isalpha() and not ch.isdigit())
    if len(digits) >= 7 and letters:
        left2 = digits[:2]
        middle3 = digits[2:5]
        right2 = digits[5:7]
        return f"{left2} {letters[0]} {middle3} ایران {right2}"

    return "نامشخص"

def detect_plate_and_ocr(image_bgr):
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
