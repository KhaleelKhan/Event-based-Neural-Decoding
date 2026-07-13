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
from policy_models import EGRUModel, EGRUNeurobenchMetricsAdapter, RNNPolicyWrapper
import torch

from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks import BenchmarkClosedLoop
from tabulate import tabulate


from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
    AverageTime,
)
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained OPS Policy Model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--neuron_file", type=str, default="gc_neuron1", help="Name of the neuron file"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=50, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--perturbed",
        action="store_true",
        help="Use perturbed environment (distribution shift)",
    )
    parser.add_argument(
        "--disturbed_fraction",
        type=float,
        default=0.2,
        help="Fraction of neurons with distribution shift (only if --perturbed)",
    )
    parser.add_argument(
        "--zeroed_fraction",
        type=float,
        default=0.1,
        help="Fraction of neurons with zeroed spikes (only if --perturbed)",
    )

    return parser.parse_args()


def benchmark_model(model, env, device, n_episodes=50, max_steps=300):
    static_metrics = [Footprint, ConnectionSparsity]
    workload_metrics = [ActivationSparsity, SynapticOperations]

    benchmark = BenchmarkClosedLoop(
        model, env, False, [], [], [static_metrics, workload_metrics]
    )
    results, avg_time = benchmark.run(
        nr_interactions=n_episodes, max_length=max_steps, device=device
    )
    results["AvgTimeToTarget"] = avg_time
    # Print results as a table

    def flatten_results(res):
        flat = {}
        for k, v in res.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    flat[f"{k}.{subk}"] = subv
            else:
                flat[k] = v
        return flat

    flat_results = flatten_results(results)
    table = [[k, flat_results[k]] for k in flat_results]
    print(
        tabulate(table, headers=["Metric", "Value"], floatfmt=".4f", tablefmt="github")
    )
    return results


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

    # Setup environment (optionally perturbed)
    if args.perturbed:
        env, device, num_neurons, max_length = setup_environment(
            neuron_file=config["neuron_file"] + ".csv",
            perturbed=True,
            disturbed_fraction=args.disturbed_fraction,
            zeroed_fraction=args.zeroed_fraction,
        )
    else:
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

    net = RNNPolicyWrapper(
        model,
        input_dim=num_neurons,
        hidden_dim=hidden_dim,
        output_dim=2,
        device=device,
    )
    neurobench_model = TorchModel(net)
    neurobench_model.add_activation_module(EGRUNeurobenchMetricsAdapter)

    # Evaluate model
    results = benchmark_model(
        neurobench_model,
        env,
        device,
        n_episodes=args.n_episodes,
        max_steps=max_length,
    )

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_copy = results.copy()

        # Add configuration info to results
        results_copy["evaluation_config"] = vars(args)
        if config:
            results_copy["training_config"] = config

        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(results_copy, f, indent=2, default=str)

        print(f"Results saved to {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
