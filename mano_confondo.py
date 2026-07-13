import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cámara - Fondo negro y mano')
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.start_button = QPushButton('Iniciar cámara')
        self.start_button.clicked.connect(self.start_camera)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        self.setLayout(layout)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convertir a fondo negro y mostrar solo la mano (detección simple por color)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Rango para color piel (ajustable)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            res = cv2.bitwise_and(frame, frame, mask=mask)
            black_bg = np.zeros_like(frame)
            hand_only = np.where(mask[..., None] != 0, res, black_bg)
            rgb_image = cv2.cvtColor(hand_only, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = CameraWidget()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
