import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContextEncoder(nn.Module):
    """Amortised inference network: maps a sequence of returns to latent z."""
    def __init__(self, input_dim=1, hidden_dim=64, latent_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=1)  # bidirectional last hidden
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class PolicyNetwork(nn.Module):
    """Policy that conditions on latent z and current state (recent returns)."""
    def __init__(self, state_dim, latent_dim, hidden_dim=64, action_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)  # output score (continuous)

    def forward(self, state, z):
        x = torch.cat([state, z], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x).squeeze(-1)  # predicted return

class AmortisedRLAgent:
    def __init__(self, state_dim=20, latent_dim=8, hidden_dim=64, lr_vae=1e-3, lr_policy=1e-4):
        self.encoder = ContextEncoder(input_dim=1, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.policy = PolicyNetwork(state_dim, latent_dim, hidden_dim, action_dim=1)
        self.opt_vae = torch.optim.Adam(self.encoder.parameters(), lr=lr_vae)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)

    def get_latent(self, context_returns):
        """context_returns: (seq_len,) array"""
        with torch.no_grad():
            x = torch.tensor(context_returns, dtype=torch.float32).view(1, -1, 1)
            mu, logvar = self.encoder(x)
            z = self.encoder.sample(mu, logvar)
        return z.squeeze(0).numpy()

    def predict_score(self, state_returns, latent_z):
        """state_returns: (state_dim,) array; latent_z: (latent_dim,) array"""
        with torch.no_grad():
            state = torch.tensor(state_returns, dtype=torch.float32).view(1, -1)
            z = torch.tensor(latent_z, dtype=torch.float32).view(1, -1)
            score = self.policy(state, z).item()
        return score

    def train_step(self, context_batch, state_batch, target_batch):
        """context_batch: (batch, seq_len, 1); state_batch: (batch, state_dim); target_batch: (batch, 1)"""
        # VAE reconstruction loss on target? Actually in PEARL, we train VAE to maximise expected return.
        # For simplicity, we train end-to-end: the policy predicts return, and we also have a KL regularisation.
        mu, logvar = self.encoder(context_batch)
        z = self.encoder.sample(mu, logvar)
        pred = self.policy(state_batch, z)
        mse_loss = F.mse_loss(pred, target_batch.squeeze(-1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / context_batch.size(0)
        total_loss = mse_loss + 0.01 * kl_loss
        self.opt_vae.zero_grad()
        self.opt_policy.zero_grad()
        total_loss.backward()
        self.opt_vae.step()
        self.opt_policy.step()
        return total_loss.item()

def prepare_meta_task(returns_df, window_days, context_len=20):
    """
    Create a meta-task: split the window into context and state/target.
    Returns: context (seq_len,), state (state_dim,), target (scalar)
    """
    n = len(returns_df)
    if n < window_days + 1:
        return None
    # Use last window_days to form context + state
    ret_series = returns_df.iloc[-window_days:].values.flatten()
    # Use first context_len days as context, next state_len days as state, and the day after as target
    # But simpler: state = recent returns, target = next day return
    if len(ret_series) < context_len + 2:
        return None
    context = ret_series[:context_len]
    state = ret_series[context_len:context_len+20]  # state_dim=20
    target = ret_series[context_len+20] if len(ret_series) > context_len+20 else None
    if target is None:
        return None
    return context, state, target
