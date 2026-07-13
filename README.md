<!--
Copyright (c) 2025 Khaleelulla Khan Nazeer and Sirine Arfa.

This file is part of Event-based Neural Decoding for Neuroprosthetic Motor Control.
See https://github.com/KhaleelKhan/Event-based-Neural-Decoding for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# 2025-BioCAS-GC-SPICE

## Cloning the Repository

Clone this repository along with its submodules:

```bash
git clone --recurse-submodules git@github.com:CityU-BRAINSys-Lab/2025-BioCAS-GC-SPICE.git
cd 2025-BioCAS-GC-SPICE
```

If you already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

We include a patched version of the `neurobench` library as a submodule, which adds support for calculating MACs for the EGRU layer.  
See the changes here: [changes](https://github.com/NeuroBench/neurobench/compare/2025_GC...KhaleelKhan:neurobench:2025_GC_EGRU)

## Installation

It is recommended to use a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then run the provided install script:

```bash
./install_requirements.sh
```

This will set up all necessary dependencies.

## Benchmarking the Models

To evaluate the trained models, use the provided `benchmark.py` script. Example usage:

```bash
python benchmark.py --model_path models/track1/gc_neuron1
```

To run with a perturbed environment (distribution shift and/or zeroed neurons):

```bash
python benchmark.py --model_path models/track2/gc_neuron1 --perturbed --disturbed_fraction 0.5 --zeroed_fraction 0.4
```

To run `benchmark.py` for all folders in `models/track1/`:

```bash
source benchmark_all.sh
```

## Rendering Trajectory Plots

To render and save trajectory plots for a trained model, use:

```bash
python render.py --model_path models/track1/gc_neuron1 --output_dir results/gc_neuron1
```


## Results


The following are the results from the benchmark script for all three neuron models across the two tracks. Track 2 uses a 50% disturbed fraction and 40% zeroed neurons. 50 evaluation trials were conducted for each model, and the average time to reach the target was recorded.

### Results from track 1

| Metric                            | Neuron 1 | Neuron 2 | Neuron 3 |
|-----------------------------------|----------|----------|----------|
| Successful trials                 | 50/50    | 50/50    | 50/50    |
| Average time taken (s)            | 0.90     | 0.87     | 0.95     |
| Footprint                         | 2172     | 2172     | 2172     |
| ConnectionSparsity                | 0.00     | 0.00     | 0.00     |
| ActivationSparsity                | 0.0006   | 0.0408   | 0.0023   |
| SynapticOperations.Effective_MACs | 4126.04  | 3947.88  | 4351.32  |
| SynapticOperations.Effective_ACs  | 5581.92  | 6492.40  | 5030.80  |
| SynapticOperations.Dense          | 38579.32 | 37426.92 | 40712.12 |
| AvgTimeToTarget                   | 0.8972   | 0.8704   | 0.9468   |

### Results from track 2

| Metric                            | Neuron 1 | Neuron 2 | Neuron 3 |
|-----------------------------------|----------|----------|----------|
| Successful trials                 | 50/50    | 50/50    | 50/50    |
| Average time taken (s)            | 1.02     | 0.84     | 1.12     |
| Footprint                         | 2172.00  | 2172.00  | 2172.00  |
| ConnectionSparsity                | 0.0000   | 0.0000   | 0.0000   |
| ActivationSparsity                | 0.0037   | 0.0505   | 0.0259   |
| SynapticOperations.Effective_MACs | 4678.64  | 3780.76  | 5100.96  |
| SynapticOperations.Effective_ACs  | 5130.48  | 3783.12  | 5446.24  |
| SynapticOperations.Dense          | 43782.32 | 35964.92 | 48099.52 |
| AvgTimeToTarget                   | 1.0182   | 0.8364   | 1.1186   |
