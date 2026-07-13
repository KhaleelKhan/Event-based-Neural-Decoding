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
import random
from pathlib import Path
import torch
from neurobench.envs.ops import OPS, OPSEnv


class PerturbedOPS(OPS):
    def __init__(
        self, *args, seed=1337, upper_lmax=100, lower_lmax=40, upper_lmin=5, **kwargs
    ):
        super().__init__(
            *args,
            upper_lmax=upper_lmax,
            lower_lmax=lower_lmax,
            upper_lmin=upper_lmin,
            **kwargs,
        )
        self.perturbation_seed = seed

        # Storing these valuesx explicitly for use in apply_perturbations
        self.upper_lmax = upper_lmax
        self.lower_lmax = lower_lmax
        self.upper_lmin = upper_lmin

    def apply_perturbations(self, disturbed_fraction=0.2, zeroed_fraction=0.0):
        torch.manual_seed(self.perturbation_seed)
        random.seed(self.perturbation_seed)

        num_neurons = self.num_neurons
        all_indices = list(range(num_neurons))

        # === Step 1: Disturbed Neurons ===
        num_disturbed = int(num_neurons * disturbed_fraction)
        disturbed_indices = random.sample(all_indices, num_disturbed)
        self.disturbed_indices = disturbed_indices

        self._pre_shift_params = {
            i: {
                "lambda_min": self.neurons[i].lambda_min,
                "lambda_max": self.neurons[i].lambda_max,
                "c": self.neurons[i].c.clone(),
            }
            for i in disturbed_indices
        }

        for i in disturbed_indices:
            c_new = torch.randn(2)
            c_new = c_new / c_new.norm()
            lambda_min_new = torch.empty(1).uniform_(0, self.upper_lmin).item()
            lambda_max_new = (
                torch.empty(1)
                .uniform_(max(lambda_min_new, self.lower_lmax), self.upper_lmax)
                .item()
            )
            self.neurons[i].assign(
                c=c_new, lambda_min=lambda_min_new, lambda_max=lambda_max_new
            )

        self._post_shift_params = {
            i: {
                "lambda_min": self.neurons[i].lambda_min,
                "lambda_max": self.neurons[i].lambda_max,
                "c": self.neurons[i].c.clone(),
            }
            for i in disturbed_indices
        }

        # === Step 2: Zeroed-out Neurons ===
        num_zeroed = int(num_neurons * zeroed_fraction)

        # Issue a warning if overlaps will occur
        # if disturbed_fraction + zeroed_fraction > 1.0:
        #     print(
        #         f"WARNING: disturbed_fraction + zeroed_fraction = {disturbed_fraction + zeroed_fraction:.2f} > 1.0.\n"
        #         f"This means some neurons will be both disturbed and zeroed."
        #     )

        zeroed_indices = random.sample(all_indices, num_zeroed)

        for i in zeroed_indices:
            self.neurons[i].removed = True



def setup_environment(
    file_path: Path | str = "thirdparty/neurobench/gc_neuron_list",
    neuron_file: str = "gc_neuron1.csv",
    perturbed: bool = False,
    disturbed_fraction: float = 0.2,
    zeroed_fraction: float = 0.1,
):
    """
    Setup OPS environment with fixed parameters.
    If perturbed=True, applies neuron perturbations.
    Returns: env, device, num_neurons, max_length
    """
    # Fixed environment parameters
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    neuron_path = file_path / neuron_file

    # device = torch.device("cpu") 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    max_duration = 3.0  # seconds
    time_step = 0.01  # seconds
    num_neurons = 96  # Number of neurons in the OPS environment

    if perturbed:
        ops = PerturbedOPS(
            num_neurons=num_neurons,
            time_step=time_step,
            upper_lmax=100,
            lower_lmax=40,
            upper_lmin=5,
            zero_prob=0.5,
            device=device,
        )
        ops.assign_neurons(neuron_path)
        ops.apply_perturbations(
            disturbed_fraction=disturbed_fraction, zeroed_fraction=zeroed_fraction
        )
    else:
        ops = OPS(
            num_neurons=96,
            time_step=time_step,
            upper_lmax=100,
            lower_lmax=40,
            upper_lmin=5,
            zero_prob=0.5,
            device=device,
        )
        ops.assign_neurons(neuron_path)

    # Initialize environment
    env = OPSEnv(
        ops=ops,
        max_duration=max_duration,
        min_time_in_target=0.5,
        side_radius=10,
        min_distance=8,
        target_size=2.5,
        device=device,
    )

    max_length = int(max_duration / time_step)

    return env, device, num_neurons, max_length


def get_env_info():
    """
    Get environment information without creating the full environment.
    Useful for model initialization.
    """
    return {
        "num_neurons": 96,
        "time_step": 0.01,
        "max_duration": 3.0,
        "min_time_in_target": 0.5,
        "target_size": 2.5,
        "action_dim": 2,  # vx, vy
    }
