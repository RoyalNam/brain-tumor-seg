import sys
import torch
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from src.config import Config
from src.model import get_model
from src.utils import load_model
from PIL import Image, ImageEnhance
from PIL.ImageQt import ImageQt
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

        self.show_border_only = False

    def initUI(self):
        self.setWindowTitle("Image Prediction")
        self.setGeometry(100, 100, 1000, 600)

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

        # Show prediction button
        show_pred_btn = QPushButton("Show Prediction", self)
        show_pred_btn.clicked.connect(self.show_prediction)
        controls_layout.addWidget(show_pred_btn)

        # Toggle border button
        self.toggle_border_btn = QPushButton("Toggle Border Only", self)
        self.toggle_border_btn.clicked.connect(self.toggle_border)
        controls_layout.addWidget(self.toggle_border_btn)

        # Add the controls layout to the main layout
        layout.addLayout(controls_layout)

        # Set main layout
        self.setLayout(layout)
    def toggle_border(self):
        # Toggle the state between showing full mask or border only
        self.show_border_only = not getattr(self, 'show_border_only', False)
        # Print the current state for debugging
        print(f"Show Border Only: {self.show_border_only}")

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
                self.pixmap = QPixmap.fromImage(q_image)

            # Display the result image in QLabel
            self.image_result.setPixmap(self.pixmap.scaled(self.image_result.width(), self.image_result.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def show_prediction(self):
        if hasattr(self, 'image') and hasattr(self, 'pixmap'):
            # Resize the mask to match the dimensions of the original image
            mask_image = self.pixmap.toImage()
            mask_image = mask_image.scaled(self.image.size[0], self.image.size[1], Qt.AspectRatioMode.KeepAspectRatio)

            # Create a transparent image with original image dimensions
            image_with_mask = self.image.convert('RGBA')

            # Access the raw bits of the QImage
            mask_bits = mask_image.bits()
            mask_bits.setsize(mask_image.width() * mask_image.height() * 4)  # Set size for RGBA format

            # Convert the raw bits to a numpy array
            mask_data = np.frombuffer(mask_bits, dtype=np.uint8).reshape((mask_image.height(), mask_image.width(), 4))

            if self.show_border_only:
                # Draw only the border of the mask (where mask pixel is non-zero)
                for y in range(1, mask_image.height() - 1):  # Avoid border pixels
                    for x in range(1, mask_image.width() - 1):  # Avoid border pixels
                        if mask_data[y, x, 0] > 0 and \
                        (mask_data[y-1, x, 0] == 0 or mask_data[y+1, x, 0] == 0 or
                            mask_data[y, x-1, 0] == 0 or mask_data[y, x+1, 0] == 0):
                            image_with_mask.putpixel((x, y), (0, 255, 0, 128))  # Semi-transparent red border
            else:
                # Draw the full mask (the whole region)
                for y in range(mask_image.height()):
                    for x in range(mask_image.width()):
                        if mask_data[y, x, 0] > 0:  # If the mask pixel is non-zero (i.e., it's part of the mask)
                            image_with_mask.putpixel((x, y), (0, 255, 0, 128))  # Semi-transparent red overlay

            # Convert the final image to QPixmap for display
            final_pixmap = QPixmap.fromImage(ImageQt(image_with_mask))
            self.image_result.setPixmap(final_pixmap.scaled(self.image_result.width(), self.image_result.height(), Qt.AspectRatioMode.KeepAspectRatio))


def main():
    app = QApplication(sys.argv)
    ex = PredictApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
