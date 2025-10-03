import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ---------------------------
# Random Fourier Features
# ---------------------------
class RFF:
    def __init__(self, input_dim, n_features=500, gamma=0.5):
        self.input_dim = input_dim
        self.n_features = n_features
        self.gamma = gamma
        self.W = np.random.normal(0, np.sqrt(2*gamma), size=(input_dim, n_features))
        self.b = np.random.uniform(0, 2*np.pi, size=n_features)

    def transform(self, X):
        projection = np.dot(X, self.W) + self.b
        return np.sqrt(2/self.n_features) * np.cos(projection)

# ---------------------------
# Linear SVM using RFF
# ---------------------------
class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# ---------------------------
# Hinge loss function
# ---------------------------
def hinge_loss(outputs, labels):
    return torch.mean(torch.clamp(1 - outputs * labels, min=0))

# ---------------------------
# SVM wrapper
# ---------------------------
class SVM:

    def __init__(self, n_features=500, gamma=0.5, lr=0.01, weight_decay=0.01):
        self.rff = None
        self.scaler = None
        self.model = None
        self.n_features = n_features
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay

    def train(self, X, y, batch_size=5000, epochs=10):
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Convert labels 0/1 to -1/+1
        y = 2*y - 1

        # Transform features using RFF
        self.rff = RFF(input_dim=X.shape[1], n_features=self.n_features, gamma=self.gamma)
        X_rff = self.rff.transform(X_scaled)
        X_rff = torch.tensor(X_rff, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1)

        # DataLoader for mini-batches
        dataset = TensorDataset(X_rff, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        self.model = LinearSVM(input_dim=self.n_features)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Training loop
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = hinge_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Last Batch Loss: {loss.item():.4f}")

    def evaluate(self, X, y):
        X_scaled = self.scaler.transform(X)
        X_rff = self.rff.transform(X_scaled)
        X_rff = torch.tensor(X_rff, dtype=torch.float32)

        y = 2*y - 1
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1)

        with torch.no_grad():
            outputs = self.model(X_rff)
            predictions = torch.sign(outputs)
            accuracy = (predictions == y_tensor).float().mean()
            print(f"Accuracy: {accuracy:.4f}")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_rff = self.rff.transform(X_scaled)
        X_rff = torch.tensor(X_rff, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(X_rff)
            predictions = (torch.sign(outputs)+1)//2  # convert -1/+1 to 0/1
        return predictions.numpy()

    def save(self, filepath):
        torch.save({
            'model_state': self.model.state_dict(),
            'rff_W': self.rff.W,
            'rff_b': self.rff.b,
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.model = LinearSVM(input_dim=self.n_features)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.rff = RFF(input_dim=self.rff.input_dim, n_features=self.n_features, gamma=self.gamma)
        self.rff.W = checkpoint['rff_W']
        self.rff.b = checkpoint['rff_b']

        self.scaler = StandardScaler()
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']
