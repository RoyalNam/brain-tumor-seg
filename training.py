import torch
from src.config import Config
from src.dataset import CustomDataset, load_annotations, load_annotation_file
from src.utils import prepare_masks, save_model
from src.model import get_model
from src.trainer import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms

config = Config()

# Load datasets
train_annotations = load_annotations(f'{config.IMAGE_DIR}/train')
valid_annotations = load_annotations(f'{config.IMAGE_DIR}/valid')
test_annotations = load_annotations(f'{config.IMAGE_DIR}/test')

train_annotation = load_annotation_file(train_annotations[0])
valid_annotation = load_annotation_file(valid_annotations[0])
test_annotation = load_annotation_file(test_annotations[0])

datasets = [
    (f'{config.IMAGE_DIR}/train', f'{config.MASK_DIR}/train', train_annotation),
    (f'{config.IMAGE_DIR}/valid', f'{config.MASK_DIR}/valid', valid_annotation),
    (f'{config.IMAGE_DIR}/test', f'{config.MASK_DIR}/test', test_annotation),
]

for img_dir, mask_dir, annotation in datasets:
    prepare_masks(img_dir, mask_dir, annotation)

# Transformations
transform = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
    transforms.Lambda(lambda x: x.clamp(0, 1))
])

# DataLoader setup
train_dataset = CustomDataset(*datasets[0], transform)
valid_dataset = CustomDataset(*datasets[1], transform)
test_dataset = CustomDataset(*datasets[2], transform)

train_loader = DataLoader(train_dataset, config.BATCH_SIZE)
valid_loader = DataLoader(valid_dataset, config.BATCH_SIZE)
test_loader = DataLoader(test_dataset, config.BATCH_SIZE)

# Model training
model = get_model(config)().to(config.DEVICE)

trainer = Trainer(model, train_loader, valid_loader, test_loader, config)
trainer.train()

test_loss = trainer.evaluate()
print('Test loss', test_loss)


save_model(model.state_dict(), 'model/model.pth')