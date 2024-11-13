import torch
from src.model import get_model
from src.config import Config
from src.utils import load_model

config = Config()

model = get_model(config)
weights = load_model('model/model.pt')
model.load_state_dict(weights)

model.eval()

def predict(input_data, device='cpu'):
    with torch.no_grad():
        input_data = input_data.to(device)
        output = model(input_data)
    return output

if __name__=='__main__':
    print('predict')
    