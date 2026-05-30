# Amortised Variational RL Engine for ETFs

Implements PEARL (Rakelly et al., 2019) / VariBAD (Zintgraf et al., 2020) – amortised variational inference for meta‑RL. The agent infers a latent variable representing the current market regime from a short history and conditions its policy on that latent, enabling rapid adaptation at inference time without retraining.

## Features
- Three ETF universes
- Seven rolling windows (63–4536 days)
- Context encoder (LSTM) maps return sequence to latent z
- Policy network conditions on state + latent to predict next‑day return
- Meta‑training across tasks (different time segments)
- Per‑ETF score = predicted return under inferred regime
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-amortised-variational-rl-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- The latent variable `z` captures the unobserved market regime.
- The policy outputs an expected return for each ETF, conditioned on both recent price action and the inferred regime.
- Higher predicted return → stronger long signal.

## Requirements

See `requirements.txt`.
