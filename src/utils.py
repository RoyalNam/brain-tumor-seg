import torch
from PIL import Image, ImageDraw
import numpy as np
import os

def prepare_masks(img_dir, mask_dir, annotation):
    print(f'{mask_dir.split("/")[-1]} masks')
    os.makedirs(mask_dir, exist_ok=True)
    totalImages = len(annotation['images'])
    done = 0
    for img, ann in zip(annotation['images'], annotation['annotations']):
        path = os.path.join(img_dir, img['file_name'])
        mask_path = os.path.join(mask_dir, img['file_name'])

        image = Image.open(path)
        segmentation = ann['segmentation']
        segmentation = np.array(segmentation[0], dtype=np.int32).reshape(-1, 2)

        mask = Image.new('L', (image.width, image.height), 0)
        draw = ImageDraw.Draw(mask)

        draw.polygon([tuple(point) for point in segmentation], outline=255, fill=255)

        mask.save(mask_path)
        
        done += 1
        print(f"{mask_dir.split('/')[-1]} {done} / {totalImages}")


def save_model(model, model_path='model/model.pt'):
    print('Save model...')
    torch.save(model, model_path)
    

def load_model(model_path):
    model = torch.load(model_path, map_location='cpu', weights_only=True)
    return model
