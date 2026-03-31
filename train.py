import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simple Model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

# Dummy Training Loop (1 Epoch for speed)
data = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
for batch_idx, (target, labels) in enumerate(data):
    target, labels = target.to(device), labels.to(device)
    optimizer.zero_grad()
    output = model(target)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx} complete.")
        break # Exit early for the activity