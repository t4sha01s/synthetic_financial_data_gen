import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import os


def compute_residuals(prices, window=30):
    log_returns = np.log(prices[1:] / prices[:-1])
    trend = pd.Series(log_returns).rolling(window).mean().fillna(method="bfill").values
    residuals = log_returns - trend
    return trend[window:], residuals[window:]

class ResidualDataset(Dataset):
    def __init__(self, prices, window=30, seq_len=30):
        trend, residuals = compute_residuals(prices, window)
        self.samples = []
        for i in range(len(residuals) - seq_len):
            cond = trend[i + seq_len - 1]
            seq = residuals[i:i + seq_len]
            self.samples.append((cond, seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cond, seq = self.samples[idx]
        return torch.tensor([cond], dtype=torch.float32), torch.tensor(seq, dtype=torch.float32)

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, seq_len, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(noise_dim + 1, hidden_dim)
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)
        self.seq_len = seq_len

    def forward(self, noise, condition):
        cond_input = torch.cat([noise, condition], dim=1)
        hidden = torch.tanh(self.fc(cond_input)).unsqueeze(0)
        c0 = torch.zeros_like(hidden)
        lstm_input = torch.zeros(noise.size(0), self.seq_len, 1)
        out, _ = self.lstm(lstm_input, (hidden, c0))
        out = self.output(out).squeeze(-1)
        return out

# Critic
class Critic(nn.Module):
    def __init__(self, seq_len, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, residuals, condition):
        x = residuals.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)
        x = torch.cat([h_n, condition], dim=1)
        return self.fc(x)

# Gradient Penalty 
def gradient_penalty(critic, real, fake, condition):
    alpha = torch.rand(real.size(0), 1, 1)
    interpolated = (alpha * real.unsqueeze(-1) + (1 - alpha) * fake.unsqueeze(-1)).requires_grad_(True)
    d_interpolated = critic(interpolated.squeeze(-1), condition)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# Метрики 
def evaluate_similarity(G, dataset, noise_dim, num_samples=500):
    G.eval()
    real_all, fake_all = [], []
    for _ in range(num_samples):
        idx = np.random.randint(len(dataset))
        condition, real = dataset[idx]
        noise = torch.randn(1, noise_dim)
        condition = condition.unsqueeze(0)
        with torch.no_grad():
            fake = G(noise, condition).numpy().flatten()
        real_all.extend(real.numpy())
        fake_all.extend(fake)

    real_all = np.array(real_all)
    fake_all = np.array(fake_all)

    wasser = wasserstein_distance(real_all, fake_all)
    mae = np.mean(np.abs(real_all - fake_all))
    rmse = np.sqrt(np.mean((real_all - fake_all) ** 2))

    return {
        "wasserstein": wasser,
        "mae": mae,
        "rmse": rmse,
        "real_mean": real_all.mean(),
        "fake_mean": fake_all.mean(),
        "real_std": real_all.std(),
        "fake_std": fake_all.std(),
    }

# ====================================== 6. Обучение ======================================
def train_wgan_gp(prices, num_epochs=2000, print_every=200, batch_size=64, seq_len=30, noise_dim=20, lambda_gp=10, save_dir="."):
    """
    Обучение WGAN-GP 
    
    Параметры:
        prices: массив цен
        num_epochs: количество эпох
        print_every: частота вывода информации
        batch_size: размер батча
        seq_len: длина последовательности
        noise_dim: размерность шума
        lambda_gp: коэффициент для gradient penalty
        save_dir: директория для сохранения результатов
    """
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics_log.csv")
    generator_path = os.path.join(save_dir, "generator.pth")
    critic_path = os.path.join(save_dir, "critic.pth")
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Saving to: {save_dir}")
    print(f"Metrics path: {metrics_path}")
    print(f"Generator path: {generator_path}")

    if os.path.exists(metrics_path):
        print("Found metrics_log.csv. Training skipped.")
        return None, None
    
    if os.path.exists(generator_path) and os.path.exists(critic_path):
        print("Found pretrained models. Loading models...")
        G = Generator(noise_dim, seq_len)
        C = Critic(seq_len)
        G.load_state_dict(torch.load(generator_path))
        C.load_state_dict(torch.load(critic_path))
        return G, C

    # Инициализация и обучение
    dataset = ResidualDataset(prices, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator(noise_dim, seq_len)
    C = Critic(seq_len)
    g_optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    c_optimizer = optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for epoch in range(1, num_epochs + 1):
        for i, (condition, real_seq) in enumerate(dataloader):
            noise = torch.randn(real_seq.size(0), noise_dim)
            fake_seq = G(noise, condition)
            real_seq = real_seq

            c_real = C(real_seq, condition)
            c_fake = C(fake_seq.detach(), condition)
            gp = gradient_penalty(C, real_seq, fake_seq, condition)
            c_loss = -(c_real.mean() - c_fake.mean()) + lambda_gp * gp

            c_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()

            if i % 5 == 0:
                noise = torch.randn(real_seq.size(0), noise_dim)
                fake_seq = G(noise, condition)
                g_loss = -C(fake_seq, condition).mean()
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        if epoch % print_every == 0:
            print(f"Epoch {epoch} | Critic loss: {c_loss.item():.4f} | Gen loss: {g_loss.item():.4f}")
            metrics = evaluate_similarity(G, dataset, noise_dim)
            print(f"Wasserstein: {metrics['wasserstein']:.4f} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f}")
            
            metrics_df = pd.DataFrame([metrics])
            metrics_df["epoch"] = epoch
            metrics_df["critic_loss"] = c_loss.item()
            metrics_df["gen_loss"] = g_loss.item()
            
            metrics_df.to_csv(metrics_path, mode='a', index=False, header=not os.path.exists(metrics_path))
            
            torch.save(G.state_dict(), generator_path)
            torch.save(C.state_dict(), critic_path)
            print(f"Models saved to {generator_path} and {critic_path}")

    torch.save(G.state_dict(), generator_path)
    torch.save(C.state_dict(), critic_path)
    print(f"Final models saved to {generator_path} and {critic_path}")
    
    return G, C

# Визуализация
def plot_saved_metrics(csv_path="metrics_log.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrics file {csv_path} not found")
    
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, df["critic_loss"], label='Critic Loss')
    plt.plot(epochs, df["gen_loss"], label='Gen Loss')
    plt.title("Losses")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, df["wasserstein"], label='Wasserstein')
    plt.plot(epochs, df["mae"], label='MAE')
    plt.plot(epochs, df["rmse"], label='RMSE')
    plt.title("Evaluation Metrics")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, df["real_mean"], label='Real Mean')
    plt.plot(epochs, df["fake_mean"], label='Fake Mean')
    plt.title("Mean of Residuals")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, df["real_std"], label='Real Std')
    plt.plot(epochs, df["fake_std"], label='Fake Std')
    plt.title("Standard Deviation of Residuals")
    plt.legend()

    plt.tight_layout()
    plt.savefig("metrics_plot.png")
    plt.show()

def plot_saved_metrics(csv_path="metrics_log.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrics file {csv_path} not found")
    
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    plt.figure(figsize=(18, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, df["critic_loss"], label='Critic Loss')
    plt.plot(epochs, df["gen_loss"], label='Gen Loss')
    plt.title("Losses")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, df["wasserstein"], label='Wasserstein')
    plt.plot(epochs, df["mae"], label='MAE')
    plt.plot(epochs, df["rmse"], label='RMSE')
    plt.title("Evaluation Metrics")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, df["real_mean"], label='Real Mean')
    plt.plot(epochs, df["fake_mean"], label='Fake Mean')
    plt.title("Mean of Residuals")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, df["real_std"], label='Real Std')
    plt.plot(epochs, df["fake_std"], label='Fake Std')
    plt.title("Standard Deviation of Residuals")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("metrics_plot.png")
    plt.show()