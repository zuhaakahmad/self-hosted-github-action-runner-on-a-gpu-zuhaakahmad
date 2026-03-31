[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/dupB1fLY)
# Self-Hosted-GitHub-Action-Runner-on-a-GPU
This activity is designed to take a "Learning by Doing" approach. It moves from the conceptual understanding of GitHub Action Runners to a hands-on implementation of a GPU-accelerated Machine Learning (ML) task.

---

## Activity: The "Neural Sprint"
**Objective:** Deploy a Self-Hosted GitHub Action Runner on a GPU-enabled Cloud Instance (AWS/Azure) to train a simple MNIST digit classifier.

### Phase 1: The Architecture Breakdown
Before touching the terminal, it is vital to understand the "Runner" concept. In GitHub Actions, the **Runner** is the compute engine that executes the steps defined in your `.yml` file. While GitHub provides hosted runners, they often lack the high-end GPUs needed for ML.

* **GitHub (The Orchestrator):** Holds the code and the workflow instructions.
* **Cloud Instance (The Muscle):** An AWS `p3.2xlarge` or Azure `Standard_NC6s_v3` acting as a "Self-Hosted Runner."
* **The Link:** A small runner application on the instance that "listens" for jobs from your GitHub repository.

---

### Phase 2: Setting up the Hardware (The "Muscle")
Choose your preferred cloud provider and launch an instance with a GPU (NVIDIA Tesla V100 or T4 are common for this).

1.  **Provision:** Launch an Ubuntu 22.04 LTS instance.
2.  **Drivers:** Ensure NVIDIA drivers and CUDA are installed.
3.  **Registration:** * Navigate to your GitHub Repo > **Settings** > **Actions** > **Runners**.
    * Click **New self-hosted runner**.
    * Follow the provided commands to download and configure the runner application on your cloud instance.
    * **Crucial:** When asked for labels, add `gpu-runner`.

---

### Phase 3: The Simple ML Task
We will use a "Hello World" of ML: training a simple Neural Network on the MNIST dataset using PyTorch.

**Create a file named `train.py` in your repo:**
```python
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
```

---

### Phase 4: The GitHub Action Workflow
Now, create the bridge. In your repo, create `.github/workflows/gpu-train.yml`.

```yaml
name: GPU Accelerated Training

on: [push]

jobs:
  train-on-gpu:
    # This targets your cloud instance specifically
    runs-on: [self-hosted, gpu-runner]

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Install Dependencies
        run: |
          pip install torch torchvision

      - name: Execute ML Task
        run: python train.py
```

---

### Phase 5: Verification
1.  **Push** your code to the repository.
2.  Go to the **Actions** tab in GitHub.
3.  Watch the logs. You should see the output: `Using device: cuda`. This confirms the GitHub Action successfully reached out to your cloud GPU to perform the computation.

> **Note:** Don't forget to shut down your AWS/Azure instances after the activity to avoid unexpected costs!
