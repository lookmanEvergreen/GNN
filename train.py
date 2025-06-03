import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import accuracy_score
from model import FakeNewsModel

# Load data
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Initialize model
model = FakeNewsModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate accuracy
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1).numpy()
    accuracy = accuracy_score(y_test.numpy(), predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/fake_news_model.pth")
print("Training complete! Model saved.")