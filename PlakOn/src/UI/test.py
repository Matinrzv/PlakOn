from pathlib import Path
from utils import load_models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = PROJECT_ROOT / "images" / "214_png.rf.aa4dc2956ae1f9cd56010672b89345e1.jpg"

_, model = load_models()
plate_text = model.predict(str(IMAGE_PATH))
print(plate_text)  # Persian text of mixed numbers and characters might not show correctly in the console
