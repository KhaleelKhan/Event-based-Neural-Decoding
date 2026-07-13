#
# Copyright (c) 2025 Khaleelulla Khan Nazeer and Sirine Arfa.
#
# This file is part of Event-based Neural Decoding for Neuroprosthetic Motor Control.
# See https://github.com/KhaleelKhan/Event-based-Neural-Decoding for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import argparse
import json
from pathlib import Path
from env_setup import setup_environment
from policy_models import EGRUModel
import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn paper style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
# matplotlib.rcParams["savefig.dpi"] = 300
# matplotlib.rcParams["figure.dpi"] = 150


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained OPS Policy Model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--neuron_file", type=str, default="gc_neuron1", help="Name of the neuron file"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=1, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save evaluation results",
    )

    return parser.parse_args()


def evaluate_model(
    model, env, device, output_dir, n_episodes=1, render=True, verbose=True
):
    """Evaluate model performance over multiple episodes"""
    model.eval()

    results = {
        "total_rewards": [],
        "final_distances": [],
        "target_reached": [],
        "time_to_target": [],
        "episode_lengths": [],
        "trajectories": [],
    }

    with torch.no_grad():
        for episode in range(n_episodes):
            state, _ = env.reset()
            state = state.to(device)
            hidden = model.init_hidden(device)
            total_reward = 0.0
            done = False

            # Track trajectory
            positions = [(env.position[0].item(), env.position[1].item())]
            targets = [(env.target[0].item(), env.target[1].item())]

            target_reached_time = None

            while not done and env.t < (env.max_duration / env.ops.time_step):
                predicted_velocity, hidden = model(
                    state.unsqueeze(0).unsqueeze(0), hidden
                )
                predicted_velocity = predicted_velocity.squeeze(0)

                # Use mean action (no exploration noise)
                action = predicted_velocity

                next_state, reward, done, _, _ = env.step(action)

                # Track when target is first reached
                current_distance = np.linalg.norm(
                    (env.position.detach() - env.target).cpu()
                )
                if target_reached_time is None and current_distance < env.target_size:
                    target_reached_time = env.t * env.ops.time_step

                total_reward += reward
                state = next_state.to(device)

                # Record position
                positions.append((env.position[0].item(), env.position[1].item()))

            # Calculate final metrics
            final_distance = np.linalg.norm((env.position.detach() - env.target).cpu())
            target_reached = final_distance < env.target_size

            # Store results
            results["total_rewards"].append(total_reward)
            results["final_distances"].append(final_distance)
            results["target_reached"].append(target_reached)
            results["time_to_target"].append(target_reached_time)
            results["episode_lengths"].append(env.t)
            results["trajectories"].append(
                {
                    "positions": positions,
                    "target": targets[0],
                    "target_size": env.target_size,
                }
            )

            status = "REACHED" if target_reached else "MISSED"
            time_str = f"{target_reached_time:.2f}s" if target_reached_time else "N/A"
            if verbose:
                print(
                    f"Episode {episode + 1:2d}: Reward={total_reward:7.2f}, "
                    f"Distance={final_distance:5.2f}, Target={status} ({time_str})"
                )

    # Calculate statistics
    success_rate = np.mean(results["target_reached"]) * 100
    avg_reward = np.mean(results["total_rewards"])
    avg_distance = np.mean(results["final_distances"])

    successful_times = [t for t in results["time_to_target"] if t is not None]
    avg_time_to_target = np.mean(successful_times) if successful_times else None

    if verbose:
        print("\n=== Evaluation Summary ===")
        print(f"Total Episodes: {n_episodes}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Final Distance: {avg_distance:.2f}")
        if avg_time_to_target:
            print(f"Average Time to Target: {avg_time_to_target:.2f}s")
        print(f"Best Reward: {max(results['total_rewards']):.2f}")
        print(f"Worst Distance: {max(results['final_distances']):.2f}")

    if render:
        render_trajectories(results, output_dir, n_episodes)

    return results


def render_trajectories(results, output_dir, n_episodes):
    """Create trajectory plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot individual trajectories
    n_cols = min(5, n_episodes)
    n_rows = (n_episodes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    # Always flatten axes to a 1D array for consistent indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    trajectories = results["trajectories"]
    for i, traj in enumerate(trajectories):
        if i >= len(axes):
            break

        ax = axes[i]

        positions = np.array(traj["positions"])
        target = traj["target"]
        target_size = traj["target_size"]

        # Plot trajectory using seaborn
        sns.lineplot(
            x=positions[:, 0],
            y=positions[:, 1],
            ax=ax,
            color="b",
            linewidth=2,
            alpha=0.7,
            label="Trajectory",
        )
        ax.plot(positions[0, 0], positions[0, 1], "go", markersize=8, label="Start")
        ax.plot(positions[-1, 0], positions[-1, 1], "ro", markersize=8, label="End")

        # Plot target
        circle = plt.Circle(
            target, target_size, color="orange", alpha=0.3, label="Target"
        )
        ax.add_patch(circle)
        ax.plot(target[0], target[1], "orange", marker="x", markersize=10)

        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_title("Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    # Hide unused subplots
    for i in range(n_episodes, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "trajectories.png", bbox_inches="tight")
    plt.close()

    print(f"Trajectory plots saved to {output_dir}")


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()

    # Load config
    config = None
    config_path = Path(args.model_path) / "config.json"
    model_path = Path(args.model_path) / "model.pth"

    config = load_config(config_path)
    print(f"Loaded config from {config_path}")

    hidden_dim = config["hidden_dim"]

    # Setup environment
    env, device, num_neurons, max_length = setup_environment(
        neuron_file=config["neuron_file"] + ".csv"
    )

    # Load model
    model = EGRUModel(num_neurons, hidden_dim, 2).to(device)

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model_state_dict = state_dict.get("model_state_dict", state_dict)
    model.load_state_dict(model_state_dict)
    print(f"Loaded model from {model_path}")

    # Evaluate model
    evaluate_model(
        model, env, device, n_episodes=args.n_episodes, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
