import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
import config
import data_manager as dm
from amortised_rl import AmortisedRLAgent, prepare_meta_task

def normalize_scores(score_dict):
    """Normalize scores to [0,1], replacing NaN/inf with 0."""
    scores = np.array(list(score_dict.values()))
    # Replace NaN and inf with 0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {k: 0.0 for k in score_dict}
    norm = (scores - min_s) / (max_s - min_s)
    return {ticker: float(norm[i]) for i, ticker in enumerate(score_dict.keys())}

def train_agent_for_window(returns_df, window_days, agent, epochs=50, batch_size=16):
    """Train the agent on tasks sampled from the rolling window."""
    tasks = []
    for start in range(0, len(returns_df) - window_days - config.CONTEXT_SIZE - 20, 5):
        segment = returns_df.iloc[start:start+window_days]
        task = prepare_meta_task(segment, window_days, context_len=config.CONTEXT_SIZE)
        if task is not None:
            tasks.append(task)
    if len(tasks) < batch_size:
        return
    context_list, state_list, target_list = [], [], []
    for ctx, st, tgt in tasks:
        context_list.append(ctx)
        state_list.append(st)
        target_list.append(tgt)
    context_t = torch.tensor(np.array(context_list), dtype=torch.float32).unsqueeze(-1)
    state_t = torch.tensor(np.array(state_list), dtype=torch.float32)
    target_t = torch.tensor(np.array(target_list), dtype=torch.float32).unsqueeze(-1)
    dataset = torch.utils.data.TensorDataset(context_t, state_t, target_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for ctx_batch, st_batch, tgt_batch in dataloader:
            loss = agent.train_step(ctx_batch, st_batch, tgt_batch)
            epoch_loss += loss
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}, loss: {epoch_loss/len(dataloader):.4f}")

def run_for_window(returns, window_days, agent=None):
    if len(returns) < window_days + config.CONTEXT_SIZE + 20:
        return None, agent
    if agent is None:
        agent = AmortisedRLAgent(state_dim=20, latent_dim=config.LATENT_DIM, hidden_dim=config.HIDDEN_SIZE,
                                 lr_vae=config.VAE_LEARNING_RATE, lr_policy=config.POLICY_LEARNING_RATE)
    train_agent_for_window(returns, window_days, agent, epochs=config.TRAIN_EPOCHS, batch_size=config.META_BATCH_SIZE)
    scores = {}
    for ticker in returns.columns:
        series = returns[ticker].values
        if len(series) < window_days:
            scores[ticker] = 0.0
            continue
        # Use last context_len days as context, last 20 days as state
        context = series[-window_days:-window_days+config.CONTEXT_SIZE] if window_days > config.CONTEXT_SIZE else series[:config.CONTEXT_SIZE]
        state = series[-20:] if len(series) >= 20 else np.pad(series, (0, 20-len(series)), constant_values=0)
        if len(context) < config.CONTEXT_SIZE:
            context = np.pad(context, (0, config.CONTEXT_SIZE - len(context)), constant_values=0)
        if len(state) < 20:
            state = np.pad(state, (0, 20 - len(state)), constant_values=0)
        latent = agent.get_latent(context)
        score = agent.predict_score(state, latent)
        # Sanitize score
        if np.isnan(score) or np.isinf(score):
            score = 0.0
        scores[ticker] = float(score)
    return scores, agent

def main():
    print("Loading master data...")
    dm.load_master_data()
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "latent_dim": config.LATENT_DIM,
        "context_size": config.CONTEXT_SIZE,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print("  No data -> skipping")
            continue
        all_window_results = []
        agent = None
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            try:
                raw_scores, agent = run_for_window(returns, w, agent)
                if raw_scores is None:
                    continue
                # Clean raw_scores again (should already be cleaned)
                raw_scores = {k: 0.0 if np.isnan(v) or np.isinf(v) else v for k, v in raw_scores.items()}
                norm_scores = normalize_scores(raw_scores)
                sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
                top_etfs = [{"ticker": t, "avi_score_norm": s, "raw_score": raw_scores[t]} for t, s in sorted_norm[:config.TOP_N]]
                all_window_results.append({
                    "window": w,
                    "top_etfs": top_etfs,
                    "all_scores_raw": raw_scores,
                    "all_scores_norm": norm_scores
                })
            except Exception as e:
                print(f"    Failed for window {w}: {e}")
        best_data = all_window_results[-1] if all_window_results else None
        results["universes"][uni_name] = {
            "best_window_data": best_data,
            "all_windows": all_window_results
        }
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/amortised_rl_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
