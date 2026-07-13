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
import torch
import torch.nn as nn
from evnn_pytorch import EGRU

class PatchedEGRU(EGRU):
    def forward(self, x, y, h):
        return self.step(x, y, h)


class EGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, egru_input_dim=4):
        super(EGRUModel, self).__init__()

        self.egru = nn.Sequential(
            nn.Linear(input_dim, egru_input_dim), 
            PatchedEGRU(egru_input_dim, hidden_dim, batch_first=True)
            )
        self.head = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Hack to make the neurobench forward hooks work with EGRU cell step
        def forward(x, y, h):
            return self.egru[1].step(x, y, h)
        self.egru[1].forward = forward

    def forward(self, x, yh=(None, None)):
        x = self.egru[0](x)
        y, h, _ = self.egru[1](x.squeeze(1), *yh)
        out = self.head(y)
        return out, (y, h)

    def init_hidden(self, device):
        return (None, None)

    def get_random_output(self, device):
        return torch.randn(1, self.output_dim, device=device)
    

class EGRUNeurobenchMetricsAdapter(nn.Identity):
    """Adapter for EGRU metrics in NeuroBench.
    This adapter is used to capture the activation sparsity of EGRU layers.

    pass the recurrent hidden state to the metrics adapter
    during the forward pass. It's output is not used.

    After TorchModel model is created, mark this class as an activation module in the NeuroBench model.

    Example:
        neurobench_model.add_activation_module(EGRUNeurobenchMetricsAdapter)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RNNPolicyWrapper(nn.Module):
    """
    Wrapper around a RNN policy model to provide an interface for NeuroBench.
    RNN states are persisted in a buffer to allow for recurrent computation.
    """

    def __init__(self, policy, input_dim, hidden_dim=128, output_dim=2, device="cpu"):
        super().__init__()
        self.policy = policy.to(device)
        self.device = device
        self.metrics_adapter = EGRUNeurobenchMetricsAdapter()
        self.reset()

        self.register_buffer(
            "data_buffer",
            torch.zeros(1, 1, input_dim, device=device).type(torch.float32),
            persistent=False,
        )

    def reset(self):
        self.hx = self.policy.init_hidden(self.device)

    def forward(self, x):
        self.data_buffer = torch.cat((self.data_buffer, x), dim=0)
        self.data_buffer = self.data_buffer[1:, :, :]

        out, self.hx = self.policy(x, self.hx)

        # Pass the recurrent output through metrics adapter to capture activation sparsity
        _ = self.metrics_adapter(self.hx[0])
        
        return out.squeeze(0)

    def init_hidden(self, device):
        return self.policy.init_hidden(self.device)
