import torch
from segmentation_models_pytorch.utils import train, losses, metrics


class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = config.DEVICE
        self.n_epochs = config.N_EPOCHS
        self.criterion = losses.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LR)
        self.metric = [metrics.IoU()]
        
        self.train_epoch = train.TrainEpoch(
            model=self.model,
            loss=self.criterion,
            metrics=self.metric,
            optimizer=self.optimizer,
            device=self.device,
        )
        
        self.valid_epoch = train.ValidEpoch(
            model=self.model,
            loss=self.criterion,
            metrics=self.metric,
            device=self.device,
        )


    def train(self):
        for epoch in range(self.n_epochs):
            train_logs = self.train_epoch.run(self.train_loader)
            valid_logs = self.valid_epoch.run(self.valid_loader)
            print(f"Epoch {epoch+1}/{self.n_epochs}")
            print(f"Train Logs: {train_logs}")
            print(f"Valid Logs: {valid_logs}")


    def evaluate(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(self.test_loader)
        print(f"Average Test Loss: {avg_test_loss}")
        return avg_test_loss
