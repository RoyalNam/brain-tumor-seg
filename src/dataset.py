import os
import glob
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, annotation, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.annotation = annotation
        self.transform = transform

    def __len__(self):
        return len(self.annotation['images'])
    
    def __getitem__(self, idx):
        img_info = self.annotation['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        mask_path = os.path.join(self.mask_dir, img_info['file_name'])
        
        img = Image.open(img_path).convert('RGB')
        img_gray = img.convert('L')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform is not None:
            img_gray = self.transform(img_gray)
            mask = self.transform(mask)
        
        return img_gray, mask

def load_annotations(image_dir):
    return glob.glob(os.path.join(image_dir, "*.json"))

def load_annotation_file(annotation_path):
    return json.load(open(annotation_path))
