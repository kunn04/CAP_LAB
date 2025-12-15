import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 28x28
        self.pool = nn.MaxPool2d(2, 2)                # 14x14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def eval_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5
    epoch_times = []
    test_accs = []

    # warmup
    xb0, _ = next(iter(train_loader))
    _ = model(xb0[:32].to(device))

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        acc = eval_accuracy(model, test_loader, device)
        t1 = time.perf_counter()
        dt = t1 - t0

        epoch_times.append(dt)
        test_accs.append(acc)
        print(f"epoch {ep:02d}/{epochs} | test_acc={acc:.4f} | epoch_time_s={dt:.4f}")

    print("FINAL: best_test_acc=", max(test_accs))
    print("FINAL: avg_epoch_time_s=", float(np.mean(epoch_times)))

    # plot
    fig, ax1 = plt.subplots()
    ax1.plot(range(1, epochs + 1), test_accs)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test accuracy")

    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), epoch_times)
    ax2.set_ylabel("Time per epoch (s)")

    plt.title(f"MNIST SmallCNN – device={device.type}")
    fig.tight_layout()
    fig.savefig("train_curve_mnist.png", dpi=200)

if __name__ == "__main__":
    main()

