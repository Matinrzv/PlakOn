import sys
import os
import psutil
from contextlib import suppress
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar, QFrame
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QImage, QFont
import cv2
from utils import load_models, normalize_plate, format_iran_plate_simple

lp_detector, lp_ocr = load_models()

class PlakOnGUI(QMainWindow):
    def __init__(self, images_folder='images'):
        super().__init__()
        self.setWindowTitle("PlakOn - پلاک خوان و مانیتورینگ")
        self.setGeometry(100, 50, 1100, 600)
        self.images_folder = images_folder
        self.image_files = os.listdir(self.images_folder)
        self.current_index = 0

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setFrameShape(QFrame.Shape.Box)
        self.image_label.setLineWidth(2)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.plate_crop_label = QLabel()
        self.plate_crop_label.setFixedSize(320, 100)
        self.plate_crop_label.setFrameShape(QFrame.Shape.Box)
        self.plate_crop_label.setLineWidth(2)
        self.plate_crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.plate_text_label = QLabel("پلاک: -")
        self.plate_text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plate_text_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.plate_text_label.setStyleSheet("color: blue;")

        self.cpu_label = QLabel("CPU:")
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setMaximum(100)

        self.ram_label = QLabel("RAM:")
        self.ram_progress = QProgressBar()
        self.ram_progress.setMaximum(100)

        self.next_btn = QPushButton("تصویر بعدی")
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setStyleSheet("font-size: 16px; padding: 8px;")

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.next_btn)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.plate_crop_label)
        right_layout.addWidget(self.plate_text_label)
        right_layout.addWidget(self.cpu_label)
        right_layout.addWidget(self.cpu_progress)
        right_layout.addWidget(self.ram_label)
        right_layout.addWidget(self.ram_progress)
        right_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_system_stats)
        self.monitor_timer.start(1000)

        self.show_next_image()

    def show_next_image(self):
        if not self.image_files:
            return
        if self.current_index >= len(self.image_files):
            self.current_index = 0

        image_path = os.path.join(self.images_folder, self.image_files[self.current_index])
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ تصویر {self.image_files[self.current_index]} یافت نشد.")
            self.current_index += 1
            return

        plate_text = "-"
        plate_cropped = None

        with suppress(Exception):
            results = lp_detector(img)[0]
            if results.boxes:
                plate = results.boxes.data.tolist()[0]
                x1, y1, x2, y2, *_ = map(int, plate[:6])
                plate_cropped = img[y1:y2, x1:x2]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                ocr_result = lp_ocr.predict(plate_cropped)
                if isinstance(ocr_result, list) and len(ocr_result) > 0:
                    plate_text = "".join([p.text if hasattr(p, "text") else str(p) for p in ocr_result])
                elif hasattr(ocr_result, "text"):
                    plate_text = ocr_result.text

                plate_text = format_iran_plate_simple(plate_text)
                self.plate_text_label.setText(f"پلاک: {plate_text}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qt_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio
        ))

        if plate_cropped is not None:
            crop_rgb = cv2.cvtColor(plate_cropped, cv2.COLOR_BGR2RGB)
            h, w, ch = crop_rgb.shape
            qt_crop = QImage(crop_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.plate_crop_label.setPixmap(QPixmap.fromImage(qt_crop).scaled(
                self.plate_crop_label.width(), self.plate_crop_label.height(), Qt.AspectRatioMode.KeepAspectRatio
            ))
        else:
            self.plate_crop_label.clear()

        self.current_index += 1

    def update_system_stats(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.cpu_progress.setValue(int(cpu))
        self.ram_progress.setValue(int(ram))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlakOnGUI()
    window.show()
    sys.exit(app.exec())
