import torch
import segmentation_models_pytorch as smp


class Config :
    IMAGE_DIR  = 'dataset/images'
    MASK_DIR = 'dataset/masks'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_EPOCHS = 10
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    LR = 0.001
    
    BACKBONE = smp.Unet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=1,
        classes=1
    )
    