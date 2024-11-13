import sys
import torch
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from src.config import Config
from src.model import get_model
from src.utils import load_model
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class PredictApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Load configuration and model
        self.config = Config()
        self.model = get_model(self.config).to(self.config.DEVICE)
        
        try:
            self.model.load_state_dict(load_model('model/model.pth'))
        except FileNotFoundError:
            print("Error: model file not found.")
            sys.exit(1)  # Exit the app if model file is missing

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
            transforms.Lambda(lambda x: x.clamp(0, 1))
        ])

    def initUI(self):
        self.setWindowTitle("Image Prediction")
        self.setGeometry(100, 100, 1000, 600)  # Change window size if necessary

        # Horizontal layout (Main layout)
        layout = QHBoxLayout()

        # Column 1: Image display and buttons
        image_layout = QVBoxLayout()

        self.image_label = QLabel(self)
        image_layout.addWidget(self.image_label)
        
        # Load button
        load_btn = QPushButton("Load Image", self)
        load_btn.clicked.connect(self.load_image)
        image_layout.addWidget(load_btn)

        # Predict button
        predict_btn = QPushButton("Predict", self)
        predict_btn.clicked.connect(self.predict)
        image_layout.addWidget(predict_btn)

        layout.addLayout(image_layout)

        # Column 2: Controls and result display
        controls_layout = QVBoxLayout()

        # Result display label
        self.image_result = QLabel(self)
        controls_layout.addWidget(self.image_result)

        # Add the controls layout to the main layout
        layout.addLayout(controls_layout)

        # Set main layout
        self.setLayout(layout)

    def load_image(self):
        # Open file dialog to select an image
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # Display the image
            self.image = Image.open(file_path).convert('RGB')
            self.display_image(file_path)

    def display_image(self, file_path):
        # Convert and display image in QLabel
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def predict(self):
        if hasattr(self, 'image'):
            # Preprocess the image and run prediction
            input_tensor = self.transform(self.image).unsqueeze(0).to(self.config.DEVICE)
            with torch.no_grad():
                output = self.model(input_tensor)
                y_pred_binary = (output > 0.5).float()
                output_image = y_pred_binary.squeeze(0).squeeze(0).cpu().numpy()

                # Print the number of 1s in the output image
                num_ones = np.sum(output_image == 1)
                print('Number of 1s in output:', num_ones)

                # Convert the output image to [0, 255] scale for display
                output_image = (output_image * 255).astype(np.uint8)

                # Convert numpy array to QImage
                height, width = output_image.shape
                q_image = QImage(output_image.data, width, height, width, QImage.Format.Format_Grayscale8)

                # Convert QImage to QPixmap
                pixmap = QPixmap.fromImage(q_image)

            # Display the result image in QLabel
            self.image_result.setPixmap(pixmap.scaled(self.image_result.width(), self.image_result.height(), Qt.AspectRatioMode.KeepAspectRatio))

            
def main():
    app = QApplication(sys.argv)
    ex = PredictApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
