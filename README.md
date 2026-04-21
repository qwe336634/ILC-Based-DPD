# ILC-Based-DPD

# DPD-RNN: Digital Predistortion with Recurrent Neural Networks

A PyTorch-based framework for Power Amplifier (PA) modeling and Digital Predistortion (DPD) using various RNN architectures.

## Overview

This project implements neural network-based Digital Predistortion for linearizing power amplifiers in wireless communication systems. It supports multiple backbone architectures for both PA modeling and DPD, with optional quantization for efficient deployment.

## Requirements

- **OS:** Ubuntu 24.04 LTS
- **Python:** 3.13
- **PyTorch:** 2.6

### Python Dependencies

```bash
pip install torch==2.6.0
pip install numpy pandas scipy matplotlib tqdm rich
pip install adabound  # Optional: for AdaBound optimizer
Features
Multiple Backbone Architectures:

GRU, DGRU, LSTM

GMP (Generalized Memory Polynomial)

PGJANET, DVRJANET

sMGU (Simplified Minimal Gated Unit)

Delta-based sparse architectures

Quantization Support: Weight and activation quantization for model compression

Comprehensive Metrics: NMSE, EVM, ACLR evaluation

Flexible Data Processing: Segment-based and frame-based dataset loading

Installation
bash
# Clone repository
git clone <repository-url>
cd dpd-rnn

# Create virtual environment (recommended)
python3.13 -m venv venv
source venv/bin/activate

# Install PyTorch 2.6
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
# or for CPU only:
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
requirements.txt
text
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.12.0
matplotlib>=3.8.0
tqdm>=4.66.0
rich>=13.0.0
Project Structure
text
.
├── backbones/          # Neural network architectures
│   ├── gru.py
│   ├── dgru.py
│   ├── lstm.py
│   ├── gmp.py
│   ├── pgjanet.py
│   ├── dvrjanet.py
│   └── smgu.py
├── modules/            # Core modules
│   ├── data_collector.py
│   ├── loggers.py
│   ├── paths.py
│   └── train_funcs.py
├── steps/              # Training/inference steps
│   ├── train_pa.py
│   ├── train_dpd.py
│   └── run_dpd.py
├── utils/              # Utility functions
│   ├── metrics.py
│   └── util.py
├── quant/              # Quantization modules
├── datasets/           # Dataset storage
├── save/              # Saved models
├── log/               # Training logs
└── dpd_out/           # DPD output signals
Quick Start
1. Prepare Dataset
Place your dataset in datasets/<dataset_name>/ with the following structure:

text
datasets/<dataset_name>/
├── train_input.csv
├── train_output.csv
├── val_input.csv
├── val_output.csv
├── test_input.csv
├── test_output.csv
└── spec.json          # Dataset specifications
Example spec.json:

json
{
    "nperseg": 16384,
    "input_signal_fs": 1000000000,
    "bw_main_ch": 100000000,
}
2. Train PA Model
bash
python main.py --step train_pa \
    --dataset_name <dataset_name> \
    --PA_backbone rvtdsmgu \
    --PA_hidden_size 8 \
    --PA_num_layers 1 \
    --frame_length 200 \
    --batch_size 256 \
    --n_epochs 100 \
    --lr 5e-4 \
    --accelerator cuda
3. Train DPD Model
bash
python main.py --step train_dpd \
    --dataset_name <dataset_name> \
    --DPD_backbone dgru \
    --DPD_hidden_size 8 \
    --DPD_num_layers 1 \
    --frame_length 200 \
    --batch_size 256 \
    --n_epochs 100 \
    --lr 5e-4
4. Run DPD Inference
bash
python main.py --step run_dpd \
    --dataset_name <dataset_name> \
    --DPD_backbone rvtdsmgu \
    --DPD_hidden_size 8
Quantization
Enable model quantization for deployment:

bash
python main.py --step train_dpd \
    --quant \
    --n_bits_w 8 \
    --n_bits_a 8 \
    --quant_dir_label q8bit \
    ...
Key Arguments
Argument	Description	Default
--step	Step to run (train_pa, train_dpd, run_dpd)	run_dpd
--dataset_name	Dataset folder name	-
--PA_backbone	PA model architecture	rvtdsmgu
--DPD_backbone	DPD model architecture	deltagru_tcnskip
--frame_length	Input frame length	200
--batch_size	Training batch size	256
--n_epochs	Number of epochs	100
--lr	Learning rate	5e-4
--hidden_size	RNN hidden size	8
--accelerator	Device (cpu, cuda, mps)	cuda
--quant	Enable quantization	False
--n_bits_w	Weight quantization bits	8
--n_bits_a	Activation quantization bits	8
Metrics
The framework evaluates models using:

NMSE: Normalized Mean Square Error (dB)

EVM: Error Vector Magnitude (dB)

ACLR: Adjacent Channel Leakage Ratio (dB) - Left, Right, and Average

Adding New Backbones
Create new backbone class in backbones/

Implement __init__ and forward methods

Add to _BACKBONE_MAP in models.py

Update argument choices in arguments.py

Example:

python
class MyBackbone(nn.Module):
    def __init__(self, hidden_size, output_size, **kwargs):
        super().__init__()
        # Define layers
    
    def forward(self, x, h_0=None):
        # Forward pass
        return output
    
    def reset_parameters(self):
        # Weight initialization
        pass
System Compatibility
Tested and verified on:

Ubuntu 24.04 LTS

Python 3.13

PyTorch 2.6
