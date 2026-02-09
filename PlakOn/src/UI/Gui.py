import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton

def on_click():
    label.setText("clicked!")
app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("PlakOn")
window.resize(400,300)
label = QLabel("Wellcome To PlakOn!",window)
label.move(50,30)
button = QPushButton("click me",window)
button.move(50,80)
button.clicked.connect(on_click)
window.show()
sys.exit(app.exec())