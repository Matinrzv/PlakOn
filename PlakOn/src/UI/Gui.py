import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout

def on_click():
    label.setText("clicked!")
app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("PlakOn")
window.resize(400,300)
label = QLabel("Wellcome To PlakOn!",window)
button = QPushButton("click me",window)
layout = QVBoxLayout()
layout.addWidget(label)
layout.addWidget(button)
window.setLayout(layout)
button.clicked.connect(on_click)
window.show()
sys.exit(app.exec())