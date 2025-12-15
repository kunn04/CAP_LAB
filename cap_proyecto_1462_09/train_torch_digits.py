import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MLP(nn.Module):
    def __init__(self, in_dim=64, hidden=128, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset (Digits) + normalización
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t, y_test_t)

    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 15
    epoch_times = []
    test_accs = []

    # warmup (para GPU/CPU )
    _ = model(X_train_t[:32].to(device))

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

        # eval
        model.eval()
        accs = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                accs.append(accuracy(logits, yb))
        acc = float(np.mean(accs))

        t1 = time.perf_counter()
        dt = t1 - t0
        epoch_times.append(dt)
        test_accs.append(acc)

        print(f"epoch {ep:02d}/{epochs} | test_acc={acc:.4f} | epoch_time_s={dt:.4f}")

    # Guardar gráfica
    fig, ax1 = plt.subplots()
    ax1.plot(range(1, epochs + 1), test_accs)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test accuracy")

    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), epoch_times)
    ax2.set_ylabel("Time per epoch (s)")

    plt.title(f"Digits MLP – device={device.type}")
    fig.tight_layout()
    fig.savefig("train_curve.png", dpi=200)

    # Resumen final
    print("FINAL: best_test_acc=", max(test_accs))
    print("FINAL: avg_epoch_time_s=", float(np.mean(epoch_times)))

if __name__ == "__main__":
    main()

