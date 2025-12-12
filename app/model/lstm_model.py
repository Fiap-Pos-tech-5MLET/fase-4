import torch
import torch.nn as nn
import os
import joblib
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class ModelTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, train_loader, epochs=10):
        self.model.train()
        loss_history = []
        for i in range(epochs):
            batch_loss = 0
            for seq, labels in train_loader:
                seq = seq.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(seq)
                
                single_loss = self.criterion(y_pred.squeeze(), labels)
                single_loss.backward()
                self.optimizer.step()
                batch_loss = single_loss.item()
            
            loss_history.append(batch_loss)

            if i % 5 == 0:
                print(f'Epoch: {i} Loss: {batch_loss:.5f}')
        return loss_history

    def evaluate(self, test_loader, scaler):
        self.model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for seq, labels in test_loader:
                seq = seq.to(self.device)
                y_pred = self.model(seq)
                predictions.append(y_pred.cpu().numpy().flatten())
                actuals.append(labels.numpy().flatten())
        
        # Concatenate and inverse scale
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        
        predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
        actuals_rescaled = scaler.inverse_transform(actuals.reshape(-1, 1))
        
        return predictions_rescaled, actuals_rescaled

    def save_model(self, path="model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="model.pth"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    # Test model instantiation
    model = LSTMModel()
    print(model)
