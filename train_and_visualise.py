#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-learning offline training with convergence visualisation.

This script:
  1. Trains a Q-table from offline flight CSV logs (or synthetic data).
  2. Generates Q-value convergence curves (P1 revision requirement).
  3. Generates gain-K variation curves from test logs (P1 revision).
  4. Supports state-space ablation (3/5/7/9 states) for P0 revision.
  5. Generates reward function comparison plots (P1 ablation).

Usage:
  # Train from synthetic data and generate convergence plots
  python train_and_visualise.py --mode train --n_states 3 --output_dir results/

  # Generate gain variation plot from a test CSV log
  python train_and_visualise.py --mode plot_gain --csv test_log.csv

  # Run state-space ablation study (synthetic)
  python train_and_visualise.py --mode ablation --output_dir results/

  # Generate all plots for the paper revision
  python train_and_visualise.py --mode all --output_dir results/
"""

import argparse
import csv
import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─── Q-learning primitives ────────────────────────────────────────────

ACTIONS = ["D", "U", "I"]  # D=increase gain, U=keep, I=decrease


def build_state_labels(n_states):
    if n_states == 3:
        return ["L", "M", "R"]
    half = (n_states - 1) // 2
    labels = []
    for i in range(half, 0, -1):
        labels.append("L" * i)
    labels.append("M")
    for i in range(1, half + 1):
        labels.append("R" * i)
    return labels


def init_Q(states, actions):
    return {s: {a: 0.0 for a in actions} for s in states}


def state_from_error(err, deadband, states):
    """Map continuous radial error to discrete state label."""
    n = len(states)
    half = (n - 1) // 2
    thresholds = []
    for i in range(half, 0, -1):
        thresholds.append(-deadband * i)
    for i in range(1, half + 1):
        thresholds.append(deadband * i)
    for i, thr in enumerate(thresholds):
        if err < thr:
            return states[i]
    return states[-1]


def reward_sparse(err_abs, e_expected, r_pos=100, r_neg=-1):
    """Sparse reward: +r_pos if within tolerance, else r_neg."""
    return r_pos if err_abs <= e_expected else r_neg


def reward_continuous(err_abs, scale=10.0):
    """Continuous reward: R = -|error| / scale (for ablation comparison)."""
    return -err_abs / scale


def epsilon_greedy(Q, s, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    vals = Q[s]
    best_val = max(vals.values())
    best_actions = [a for a, v in vals.items() if v == best_val]
    return random.choice(best_actions)


# ─── Synthetic environment ────────────────────────────────────────────

class SimpleSpiralEnv:
    """Simplified 1-D radial-error dynamics for offline training.

    The UAV tracks a spiral path; wind and initial conditions introduce
    a radial deviation.  The controller adjusts gain K which affects
    how quickly the error is reduced.
    """

    def __init__(self, wind_speed=0.0, dt=0.2, noise_std=0.5):
        self.wind_speed = wind_speed
        self.dt = dt
        self.noise_std = noise_std
        self.K = 1.0
        self.error = 0.0

    def reset(self, initial_error=None):
        if initial_error is None:
            self.error = random.uniform(-15, 15)
        else:
            self.error = initial_error
        self.K = 1.0
        return self.error

    def step(self, action, K_step=0.10):
        # Apply action to gain
        if action == "D":
            self.K += K_step
        elif action == "I":
            self.K -= K_step
        self.K = max(0.5, min(2.0, self.K))

        # Simplified error dynamics: error decays by K-dependent rate,
        # wind adds a bias, and there is stochastic noise
        correction = -self.K * self.error * 0.15 * self.dt
        wind_effect = self.wind_speed * 0.3 * self.dt * math.sin(random.uniform(0, 2 * math.pi))
        noise = random.gauss(0, self.noise_std) * self.dt

        self.error += correction + wind_effect + noise
        return self.error


# ─── Training ─────────────────────────────────────────────────────────

def train_qtable(
    n_states=3,
    deadband=5.0,
    e_expected=5.0,
    alpha=0.01,
    gamma=0.8,
    epsilon_start=1.0,
    epsilon_end=0.01,
    episodes=1000,
    steps_per_episode=200,
    wind_speeds=None,
    reward_type="sparse",
    K_step=0.10,
    seed=42,
):
    """Train Q-table and record convergence data.

    Returns:
        Q: dict of state→action→value
        history: dict with keys 'episode_rewards', 'q_max_per_episode',
                 'q_delta_per_episode'
    """
    random.seed(seed)
    states = build_state_labels(n_states)
    Q = init_Q(states, ACTIONS)

    if wind_speeds is None:
        wind_speeds = [0, 2, 4, 6, 8, 10]

    history = {
        "episode_rewards": [],
        "q_max_per_episode": [],
        "q_delta_per_episode": [],
    }

    for ep in range(episodes):
        ws = random.choice(wind_speeds)
        env = SimpleSpiralEnv(wind_speed=ws)
        err = env.reset()
        total_reward = 0.0

        # Epsilon decay
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep / max(1, episodes - 1))

        # Snapshot Q before episode
        q_before = {s: dict(Q[s]) for s in states}

        for _ in range(steps_per_episode):
            s = state_from_error(err, deadband, states)
            a = epsilon_greedy(Q, s, epsilon)
            err_next = env.step(a, K_step=K_step)

            if reward_type == "sparse":
                r = reward_sparse(abs(err_next), e_expected)
            else:
                r = reward_continuous(abs(err_next))

            s_next = state_from_error(err_next, deadband, states)

            # Q-learning update
            best_next = max(Q[s_next].values())
            Q[s][a] += alpha * (r + gamma * best_next - Q[s][a])

            total_reward += r
            err = err_next

        # Record convergence metrics
        history["episode_rewards"].append(total_reward)

        q_max = max(Q[s][a] for s in states for a in ACTIONS)
        history["q_max_per_episode"].append(q_max)

        delta = max(
            abs(Q[s][a] - q_before[s][a])
            for s in states
            for a in ACTIONS
        )
        history["q_delta_per_episode"].append(delta)

    return Q, history, states


# ─── Plotting helpers ─────────────────────────────────────────────────

def plot_convergence(history, output_path, title_suffix=""):
    """Generate Q-value convergence curve (P1 revision Fig. requirement)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    eps = range(1, len(history["episode_rewards"]) + 1)

    # (a) Cumulative reward per episode
    axes[0].plot(eps, history["episode_rewards"], linewidth=0.5, alpha=0.5, color="steelblue")
    # Smoothed
    window = 20
    if len(history["episode_rewards"]) > window:
        smoothed = np.convolve(history["episode_rewards"],
                               np.ones(window) / window, mode="valid")
        axes[0].plot(range(window, len(history["episode_rewards"]) + 1),
                     smoothed, linewidth=2, color="darkblue", label=f"Moving avg ({window} eps)")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title(f"Q-learning Training Convergence{title_suffix}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) Max Q-value
    axes[1].plot(eps, history["q_max_per_episode"], linewidth=1, color="darkorange")
    axes[1].set_ylabel("Max Q-value")
    axes[1].grid(True, alpha=0.3)

    # (c) Q-value change per episode
    axes[2].plot(eps, history["q_delta_per_episode"], linewidth=0.5, color="forestgreen")
    axes[2].set_ylabel("Max ΔQ per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot: {output_path}")


def plot_gain_history(csv_path, output_path):
    """Plot gain K variation over flight time (P1 revision)."""
    times, gains = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_s"]))
            gains.append(float(row["K"]))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, gains, linewidth=1.2, color="royalblue")
    ax.set_xlabel("Flight Time (s)")
    ax.set_ylabel("Gain K")
    ax.set_title("Adaptive Gain K Variation During Flight")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved gain history plot: {output_path}")


def plot_gain_from_main_csv(csv_path, output_path):
    """Extract gain K and altitude error from the main test CSV log."""
    times, gains, alt_errors, dr_values, progress_vals = [], [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_s"]))
            gains.append(float(row["K"]))
            if "alt_error_m" in row and row["alt_error_m"]:
                alt_errors.append(float(row["alt_error_m"]))
            dr_values.append(float(row["abs_dr_m"]))
            if "progress" in row:
                progress_vals.append(float(row["progress"]))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(times, gains, linewidth=1, color="royalblue")
    axes[0].set_ylabel("Gain K")
    axes[0].set_title("Q-learning Adaptive Control During Flight")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, dr_values, linewidth=0.8, color="crimson", alpha=0.7)
    axes[1].axhline(y=5.0, color="gray", linestyle="--", alpha=0.5, label="|dr| = 5m threshold")
    axes[1].set_ylabel("|dr| Radial Error (m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if alt_errors:
        axes[2].plot(times[:len(alt_errors)], [abs(e) for e in alt_errors],
                     linewidth=0.8, color="forestgreen", alpha=0.7)
        axes[2].set_ylabel("|Altitude Error| (m)")
    axes[2].set_xlabel("Flight Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved control analysis plot: {output_path}")


def plot_ablation_results(results, output_path):
    """Bar chart comparing state-space granularity ablation (P0 revision)."""
    n_states_list = sorted(results.keys())
    mae_vals = [results[n]["mae"] for n in n_states_list]
    convergence_eps = [results[n]["convergence_episode"] for n in n_states_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = axes[0].bar([str(n) for n in n_states_list], mae_vals, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Number of Discrete States")
    axes[0].set_ylabel("Training MAE (m)")
    axes[0].set_title("State-Space Granularity Ablation: Tracking Error")
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, mae_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    bars2 = axes[1].bar([str(n) for n in n_states_list], convergence_eps,
                        color="darkorange", alpha=0.8)
    axes[1].set_xlabel("Number of Discrete States")
    axes[1].set_ylabel("Convergence Episode")
    axes[1].set_title("State-Space Granularity Ablation: Convergence Speed")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, convergence_eps):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(val), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved ablation plot: {output_path}")


def plot_reward_ablation(sparse_history, continuous_history, output_path):
    """Compare sparse vs continuous reward training curves (P1 revision)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    eps_s = range(1, len(sparse_history["episode_rewards"]) + 1)
    eps_c = range(1, len(continuous_history["episode_rewards"]) + 1)

    window = 20

    # Cumulative reward
    for ax_idx, (label, hist, eps, color) in enumerate([
        ("Sparse Reward (R∈{100, -1})", sparse_history, eps_s, "steelblue"),
        ("Continuous Reward (R=-|e|/10)", continuous_history, eps_c, "darkorange"),
    ]):
        axes[0].plot(eps, hist["episode_rewards"], linewidth=0.3, alpha=0.3, color=color)
        if len(hist["episode_rewards"]) > window:
            smoothed = np.convolve(hist["episode_rewards"],
                                   np.ones(window) / window, mode="valid")
            axes[0].plot(range(window, len(hist["episode_rewards"]) + 1),
                         smoothed, linewidth=2, color=color, label=label)

    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title("Reward Function Ablation: Training Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q-value convergence
    axes[1].plot(eps_s, sparse_history["q_delta_per_episode"],
                 linewidth=0.5, color="steelblue", label="Sparse")
    axes[1].plot(eps_c, continuous_history["q_delta_per_episode"],
                 linewidth=0.5, color="darkorange", label="Continuous")
    axes[1].set_ylabel("Max ΔQ per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved reward ablation plot: {output_path}")


# ─── Ablation runner ──────────────────────────────────────────────────

def run_state_ablation(output_dir):
    """Run state-space granularity ablation (3, 5, 7, 9 states)."""
    results = {}
    for n in [3, 5, 7, 9]:
        Q, hist, states = train_qtable(n_states=n, episodes=1000)

        # Evaluate: run a few test episodes with greedy policy
        env = SimpleSpiralEnv(wind_speed=5.0)
        errors = []
        for _ in range(50):
            err = env.reset()
            for _ in range(200):
                s = state_from_error(err, 5.0, states)
                vals = Q[s]
                best_val = max(vals.values())
                best_a = [a for a, v in vals.items() if v == best_val]
                a = best_a[0]
                err = env.step(a)
                errors.append(abs(err))

        mae = sum(errors) / len(errors)

        # Find convergence episode (delta < 0.01)
        conv_ep = len(hist["q_delta_per_episode"])
        for i, d in enumerate(hist["q_delta_per_episode"]):
            if d < 0.01:
                conv_ep = i + 1
                break

        results[n] = {"mae": mae, "convergence_episode": conv_ep, "Q": Q, "states": states}

        # Save Q-table
        qtable_path = os.path.join(output_dir, f"qtable_{n}state.json")
        with open(qtable_path, "w") as f:
            json.dump({
                "states": states,
                "actions": ACTIONS,
                "n_states": n,
                "Q": Q,
            }, f, indent=2)
        print(f"  {n}-state: MAE={mae:.3f}m  convergence@ep{conv_ep}  → {qtable_path}")

        # Save convergence plot
        plot_convergence(
            hist,
            os.path.join(output_dir, f"convergence_{n}state.png"),
            title_suffix=f" ({n} states)",
        )

    return results


def run_reward_ablation(output_dir):
    """Compare sparse vs continuous reward (P1 revision)."""
    print("\n=== Reward Function Ablation ===")
    _, hist_sparse, _ = train_qtable(reward_type="sparse", episodes=1000)
    _, hist_cont, _ = train_qtable(reward_type="continuous", episodes=1000)
    plot_reward_ablation(
        hist_sparse, hist_cont,
        os.path.join(output_dir, "reward_ablation.png"),
    )


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Q-learning training and visualisation.")
    ap.add_argument("--mode", choices=["train", "plot_gain", "ablation", "all"],
                    default="all")
    ap.add_argument("--n_states", type=int, default=3, choices=[3, 5, 7, 9])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--csv", default="", help="Path to test log CSV for gain plotting")
    ap.add_argument("--output_dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ("train", "all"):
        print("=== Training Q-table ===")
        Q, hist, states = train_qtable(
            n_states=args.n_states,
            episodes=args.episodes,
        )

        # Save Q-table
        qtable_path = os.path.join(args.output_dir, f"qtable_trained_{args.n_states}state.json")
        with open(qtable_path, "w") as f:
            json.dump({
                "states": states,
                "actions": ACTIONS,
                "n_states": args.n_states,
                "Q": Q,
            }, f, indent=2)
        print(f"Saved Q-table: {qtable_path}")
        print(f"Learned policy: { {s: max(Q[s], key=Q[s].get) for s in states} }")

        # Convergence plot
        plot_convergence(
            hist,
            os.path.join(args.output_dir, f"convergence_{args.n_states}state.png"),
        )

    if args.mode == "plot_gain":
        if not args.csv:
            raise SystemExit("--csv required for plot_gain mode")
        # Try gain history CSV first, then main CSV
        gain_csv = args.csv.replace(".csv", "_gain_history.csv")
        if os.path.exists(gain_csv):
            plot_gain_history(gain_csv, os.path.join(args.output_dir, "gain_variation.png"))
        if os.path.exists(args.csv):
            plot_gain_from_main_csv(args.csv, os.path.join(args.output_dir, "control_analysis.png"))

    if args.mode in ("ablation", "all"):
        print("\n=== State-Space Ablation ===")
        results = run_state_ablation(args.output_dir)
        plot_ablation_results(results, os.path.join(args.output_dir, "state_ablation.png"))

        run_reward_ablation(args.output_dir)

    print("\nDone. All outputs in:", args.output_dir)


if __name__ == "__main__":
    main()
