import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator

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
class SVM(BaseEstimator):

    def __init__(self, n_features=500, gamma=0.5, lr=0.01, weight_decay=0.01, verbose=0):
        self.rff = None
        self.scaler = None
        self.model = None
        self.n_features = n_features
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

    def fit(self, X, y):
        self.train(X, y)

    def train(self, X, y, batch_size=5000, epochs=10):
        if self.gamma == "scale":
            self.gamma = 1 / X.shape[1]
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Convert labels 0/1 to -1/+1
        y = np.array(y)
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
            if self.verbose > 0:
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

    def predict_proba(self, X):
        """
        Predict class probabilities for new data X.
        
        Returns:
            probs: numpy array of shape (n_samples, 2), where
                probs[:, 0] = P(class 0), probs[:, 1] = P(class 1)
        """
        # 1. Scale the input
        X_scaled = self.scaler.transform(X)

        # 2. Transform using RFF
        X_rff = self.rff.transform(X_scaled)
        X_rff = torch.tensor(X_rff, dtype=torch.float32)

        # 3. Get raw SVM output
        with torch.no_grad():
            outputs = self.model(X_rff)

        # 4. Map to probability via sigmoid
        probs_pos = torch.sigmoid(outputs).numpy().flatten()  # P(class 1)
        probs_neg = 1 - probs_pos                               # P(class 0)

        # 5. Stack as (n_samples, 2)
        probs = np.vstack([probs_neg, probs_pos]).T
        return probs

    def save(self, filepath):
        """
        Save the model weights, RFF parameters, and scaler statistics
        """
        torch.save({
            'model_state': self.model.state_dict(),   # model weights
            'rff_W': self.rff.W,                      # RFF weight matrix
            'rff_b': self.rff.b,                      # RFF bias vector
            'scaler_mean': self.scaler.mean_,         # StandardScaler mean
            'scaler_scale': self.scaler.scale_        # StandardScaler scale
        }, filepath)


    def load(self, filepath):
        checkpoint = torch.load(filepath, weights_only=False)

        # Use saved RFF shape to set model input/output
        input_dim = checkpoint['rff_W'].shape[0]   # original input dimension
        n_features = checkpoint['rff_W'].shape[1]  # number of RFF features
        self.n_features = n_features

        # Reconstruct model
        self.model = LinearSVM(input_dim=self.n_features)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        # Reconstruct RFF
        self.rff = RFF(input_dim=input_dim, n_features=n_features, gamma=self.gamma)
        self.rff.W = checkpoint['rff_W']
        self.rff.b = checkpoint['rff_b']

        # Reconstruct scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']
