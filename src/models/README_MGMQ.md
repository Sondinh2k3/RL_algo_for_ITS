# MGMQ-PPO: Multi-agent Graph-based Multi-scale Q-learning with PPO

This module implements the MGMQ architecture for traffic signal control, adapted to use PPO (Proximal Policy Optimization) instead of Q-learning for continuous action spaces.

## Architecture Overview

The MGMQ architecture follows this flow (as shown in the diagram):

```
State (Intersections) 
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                         MGMQ                                 │
│  ┌─────────┐     ┌──────────────┐                           │
│  │   GAT   │────▶│ GraphSAGE    │                           │
│  │         │     │  + Bi-GRU    │                           │
│  └─────────┘     └──────────────┘                           │
│  Intersection        Network      ──▶ ⊕ ──▶ Q-Network/PPO  │
│   Embedding         Embedding         │     (Actor-Critic)  │
│                                  Joint Embedding             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Action (Signal Light: 🔴🟡🟢)
```

### Components

1. **GAT (Graph Attention Network)**: Encodes intersection features using attention mechanism to learn which neighboring intersections are most relevant.

2. **GraphSAGE + Bi-GRU**: 
   - GraphSAGE aggregates neighborhood information
   - Bi-GRU captures temporal/sequential patterns across the network

3. **Joint Embedding**: Concatenates intersection and network embeddings

4. **PPO Actor-Critic**: Outputs continuous actions for traffic signal control

## Files Structure

```
src/models/
├── __init__.py           # Module exports
├── gat_layer.py          # GAT layer implementation
├── graphsage_bigru.py    # GraphSAGE + Bi-GRU implementation
└── mgmq_model.py         # Complete MGMQ model with RLlib integration

scripts/
├── train_mgmq_ppo.py     # Training script
└── eval_mgmq_ppo.py      # Evaluation script
```

## Usage

### Training

```bash
# Basic training on grid4x4 network
python scripts/train_mgmq_ppo.py --network grid4x4 --iterations 200

# With custom MGMQ hyperparameters
python scripts/train_mgmq_ppo.py \
    --network grid4x4 \
    --iterations 500 \
    --workers 4 \
    --gat-hidden-dim 64 \
    --gat-output-dim 32 \
    --gat-num-heads 4 \
    --graphsage-hidden-dim 64 \
    --gru-hidden-dim 32 \
    --learning-rate 3e-4 \
    --gpu

# With early stopping
python scripts/train_mgmq_ppo.py \
    --network grid4x4 \
    --iterations 1000 \
    --reward-threshold -100 \
    --patience 50
```

### Evaluation

```bash
# Evaluate a trained model
python scripts/eval_mgmq_ppo.py \
    --checkpoint ./results_mgmq/experiment_name/checkpoint_xxx \
    --network grid4x4 \
    --episodes 10 \
    --output results.json

# With SUMO GUI
python scripts/eval_mgmq_ppo.py \
    --checkpoint ./results_mgmq/experiment_name/checkpoint_xxx \
    --network grid4x4 \
    --gui
```

## Model Configuration

### MGMQ Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gat_hidden_dim` | 64 | Hidden dimension for GAT input projection |
| `gat_output_dim` | 32 | Output dimension per GAT attention head |
| `gat_num_heads` | 4 | Number of GAT attention heads |
| `graphsage_hidden_dim` | 64 | Hidden dimension for GraphSAGE |
| `gru_hidden_dim` | 32 | Hidden dimension for Bi-GRU |
| `policy_hidden_dims` | [128, 64] | Hidden layers for policy network |
| `value_hidden_dims` | [128, 64] | Hidden layers for value network |
| `dropout` | 0.3 | Dropout rate |

### PPO Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `lambda_` | 0.95 | GAE lambda |
| `entropy_coeff` | 0.01 | Entropy coefficient |
| `clip_param` | 0.2 | PPO clip parameter |

## Key Differences from Original MGMQ

1. **PPO instead of Q-learning**: The original MGMQ uses Q-learning which is not suitable for continuous action spaces. We use PPO with actor-critic architecture.

2. **Environment unchanged**: State space, action space, and reward function remain the same as your current project.

3. **GNN layers added**: The GAT and GraphSAGE+Bi-GRU layers are added before the policy/value networks.

## Network Adjacency

The model automatically builds an adjacency matrix based on traffic signal IDs:
- For grid networks (e.g., "A0", "A1", "B0"): Infers grid connectivity
- For custom networks: Can parse SUMO network files

## Requirements

- PyTorch >= 1.12
- Ray[rllib] <= 2.9.0
- gymnasium >= 1.1.1
- pettingzoo >= 1.25.0

## References

- Graph Attention Networks: Veličković et al., ICLR 2018
- GraphSAGE: Hamilton et al., NeurIPS 2017
- MGMQ: Multi-agent Graph-based Multi-scale Q-learning
- PPO: Schulman et al., 2017
