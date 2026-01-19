"""
Test GAT and GraphSAGE+BiGRU Forward Pass with SUMO-like Inputs.

This test module verifies that observations from SUMO environment can
successfully pass through the MGMQ model's GAT and GraphSAGE+BiGRU layers.

SUMO observation format:
- Per intersection: 4 features x 12 detectors (lanes) = 48 features
- Features: density, queue, occupancy, average_speed (each normalized to [0, 1])
- Multi-agent: Dictionary of {agent_id: observation_array}

MGMQ Architecture:
1. Input: [batch, num_agents, obs_dim] or [batch, obs_dim] for single agent
2. GAT Layer: Lane-level attention (12 lanes per intersection)
3. GraphSAGE + BiGRU: Network-level embedding
4. Output: Joint embedding for policy/value networks

Author: Son Dinh
Date: 2025
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.gat_layer import GATLayer, MultiHeadGATLayer, get_lane_conflict_matrix, get_lane_cooperation_matrix
from src.models.graphsage_bigru import GraphSAGE_BiGRU, TemporalGraphSAGE_BiGRU, GraphSAGELayer
from src.models.mgmq_model import MGMQEncoder, MGMQModel, build_network_adjacency


class TestSUMOObservationFormat(unittest.TestCase):
    """Test that SUMO observation format is compatible with MGMQ model."""
    
    def setUp(self):
        """Set up test fixtures with SUMO-like observation dimensions."""
        # SUMO observation format: 4 features * 12 detectors = 48
        self.num_detectors = 12  # Standard 4-way intersection with 3 lanes per direction
        self.num_features_per_detector = 4  # density, queue, occupancy, speed
        self.obs_dim = self.num_detectors * self.num_features_per_detector  # 48
        
        # Multi-agent setup (grid4x4 has 16 intersections)
        self.num_agents = 16
        self.batch_size = 4
        
        # Action space (typically 4-8 phases)
        self.action_dim = 4
        
    def _create_sumo_like_observation(
        self, 
        batch_size: int, 
        num_agents: int,
        obs_dim: int,
        as_dict: bool = False
    ) -> torch.Tensor:
        """
        Create SUMO-like observations with realistic value ranges.
        
        Args:
            batch_size: Number of samples
            num_agents: Number of traffic signals
            obs_dim: Observation dimension per agent
            as_dict: If True, return dict format like SUMO env
            
        Returns:
            Tensor of shape [batch, num_agents, obs_dim] or dict
        """
        # Generate normalized observations in [0, 1]
        # Mimic SUMO observation structure: density, queue, occupancy, speed
        obs = torch.rand(batch_size, num_agents, obs_dim)
        
        # Clip to [0, 1] as done in SUMO environment
        obs = torch.clamp(obs, 0.0, 1.0)
        
        if as_dict:
            # Convert to dict format like SUMO env returns
            ts_ids = [f"ts_{i}" for i in range(num_agents)]
            obs_dict = {}
            for i, ts_id in enumerate(ts_ids):
                obs_dict[ts_id] = obs[:, i, :].numpy()
            return obs_dict
        
        return obs
    
    def test_observation_dimensions(self):
        """Test that observations have correct dimensions for MGMQ."""
        obs = self._create_sumo_like_observation(
            self.batch_size, self.num_agents, self.obs_dim
        )
        
        # Verify shape
        self.assertEqual(obs.shape, (self.batch_size, self.num_agents, self.obs_dim))
        
        # Verify values are in [0, 1]
        self.assertTrue(torch.all(obs >= 0))
        self.assertTrue(torch.all(obs <= 1))
        
    def test_observation_reshaping_for_lanes(self):
        """Test that observations can be reshaped to lane format for GAT."""
        obs = self._create_sumo_like_observation(
            self.batch_size, self.num_agents, self.obs_dim
        )
        
        # Reshape to [batch * num_agents, 12 lanes, features_per_lane]
        obs_flat = obs.view(-1, self.obs_dim)  # [batch * agents, obs_dim]
        lane_features = obs_flat.view(-1, self.num_detectors, self.num_features_per_detector)
        
        expected_shape = (
            self.batch_size * self.num_agents, 
            self.num_detectors, 
            self.num_features_per_detector
        )
        self.assertEqual(lane_features.shape, expected_shape)


class TestGATLayerWithSUMOInput(unittest.TestCase):
    """Test GAT layer with SUMO-like inputs."""
    
    def setUp(self):
        """Set up GAT layer and test parameters."""
        self.num_lanes = 12
        self.in_features = 4  # Features per lane (density, queue, occupancy, speed)
        self.gat_hidden_dim = 32
        self.gat_output_dim = 16
        self.gat_num_heads = 4
        self.batch_size = 4
        
    def test_single_head_gat_forward(self):
        """Test single-head GAT layer forward pass."""
        gat = GATLayer(
            in_features=self.gat_hidden_dim,
            out_features=self.gat_output_dim,
            dropout=0.0,  # Disable dropout for testing
            concat=True
        )
        
        # Project input to hidden dim first
        input_proj = nn.Linear(self.in_features, self.gat_hidden_dim)
        
        # Create SUMO-like lane features
        lane_features = torch.rand(self.batch_size, self.num_lanes, self.in_features)
        lane_features = torch.clamp(lane_features, 0.0, 1.0)
        
        # Project to hidden dimension
        h = input_proj(lane_features)  # [batch, 12, gat_hidden_dim]
        
        # Get adjacency matrix (cooperation)
        adj = get_lane_cooperation_matrix()  # [12, 12]
        adj_batch = adj.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        # Forward pass
        gat.eval()  # Set to eval mode to disable dropout
        with torch.no_grad():
            output = gat(h, adj_batch)
        
        # Verify output shape
        expected_shape = (self.batch_size, self.num_lanes, self.gat_output_dim)
        self.assertEqual(output.shape, expected_shape)
        
        # Verify no NaN values
        self.assertFalse(torch.isnan(output).any(), "GAT output contains NaN values")
        
    def test_multi_head_gat_forward(self):
        """Test multi-head GAT layer forward pass."""
        gat = MultiHeadGATLayer(
            in_features=self.gat_hidden_dim,
            out_features=self.gat_output_dim,
            n_heads=self.gat_num_heads,
            dropout=0.0,
            concat=True
        )
        
        # Project input to hidden dim first
        input_proj = nn.Linear(self.in_features, self.gat_hidden_dim)
        
        # Create SUMO-like lane features
        lane_features = torch.rand(self.batch_size, self.num_lanes, self.in_features)
        
        # Project to hidden dimension
        h = input_proj(lane_features)
        
        # Get adjacency matrices
        adj_coop = get_lane_cooperation_matrix().unsqueeze(0).expand(self.batch_size, -1, -1)
        adj_conf = get_lane_conflict_matrix().unsqueeze(0).expand(self.batch_size, -1, -1)
        
        # Forward pass with cooperation matrix
        gat.eval()
        with torch.no_grad():
            output_coop = gat(h, adj_coop)
            output_conf = gat(h, adj_conf)
        
        # Verify output shape (concat mode: output_dim * n_heads)
        expected_shape = (self.batch_size, self.num_lanes, self.gat_output_dim * self.gat_num_heads)
        self.assertEqual(output_coop.shape, expected_shape)
        self.assertEqual(output_conf.shape, expected_shape)
        
        # Verify no NaN values
        self.assertFalse(torch.isnan(output_coop).any(), "GAT cooperation output contains NaN")
        self.assertFalse(torch.isnan(output_conf).any(), "GAT conflict output contains NaN")
        
    def test_gat_with_different_adjacency_matrices(self):
        """Test GAT produces different outputs with different adjacency matrices."""
        gat = MultiHeadGATLayer(
            in_features=self.gat_hidden_dim,
            out_features=self.gat_output_dim,
            n_heads=self.gat_num_heads,
            dropout=0.0,
            concat=True
        )
        
        input_proj = nn.Linear(self.in_features, self.gat_hidden_dim)
        lane_features = torch.rand(self.batch_size, self.num_lanes, self.in_features)
        h = input_proj(lane_features)
        
        adj_coop = get_lane_cooperation_matrix().unsqueeze(0).expand(self.batch_size, -1, -1)
        adj_conf = get_lane_conflict_matrix().unsqueeze(0).expand(self.batch_size, -1, -1)
        
        gat.eval()
        with torch.no_grad():
            output_coop = gat(h, adj_coop)
            output_conf = gat(h, adj_conf)
        
        # Outputs should be different due to different adjacency structures
        # (unless the matrices are identical, which they shouldn't be)
        self.assertFalse(
            torch.allclose(output_coop, output_conf),
            "GAT outputs should differ for cooperation vs conflict adjacency"
        )


class TestGraphSAGEBiGRUWithSUMOInput(unittest.TestCase):
    """Test GraphSAGE+BiGRU layer with SUMO-like inputs."""
    
    def setUp(self):
        """Set up GraphSAGE+BiGRU and test parameters."""
        self.num_agents = 16  # grid4x4
        self.in_features = 128  # Output from GAT (gat_output_dim * heads * 2)
        self.graphsage_hidden_dim = 64
        self.gru_hidden_dim = 32
        self.batch_size = 4
        
    def test_graphsage_layer_forward(self):
        """Test GraphSAGE layer forward pass."""
        graphsage = GraphSAGELayer(
            in_features=self.in_features,
            out_features=self.graphsage_hidden_dim,
            aggregator_type='mean',
            dropout=0.0
        )
        
        # Create input (after GAT processing)
        # Shape: [batch, num_agents, in_features]
        node_features = torch.rand(self.batch_size, self.num_agents, self.in_features)
        
        # Create simple adjacency (grid structure with neighbors)
        adj = torch.eye(self.num_agents)
        # Add connections between neighboring intersections in 4x4 grid
        for i in range(self.num_agents):
            if i % 4 != 3:  # Right neighbor
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
            if i < 12:  # Bottom neighbor
                adj[i, i + 4] = 1
                adj[i + 4, i] = 1
        adj = adj.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        # Forward pass
        graphsage.eval()
        with torch.no_grad():
            output = graphsage(node_features, adj)
        
        expected_shape = (self.batch_size, self.num_agents, self.graphsage_hidden_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any(), "GraphSAGE output contains NaN")
        
    def test_graphsage_bigru_forward(self):
        """Test combined GraphSAGE+BiGRU forward pass."""
        model = GraphSAGE_BiGRU(
            in_features=self.in_features,
            hidden_features=self.graphsage_hidden_dim,
            gru_hidden_size=self.gru_hidden_dim,
            dropout=0.0
        )
        
        # Create input
        node_features = torch.rand(self.batch_size, self.num_agents, self.in_features)
        adj = torch.eye(self.num_agents)
        for i in range(self.num_agents):
            if i % 4 != 3:
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
            if i < 12:
                adj[i, i + 4] = 1
                adj[i + 4, i] = 1
        adj = adj.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        # Forward pass (GraphSAGE only - returns sequence)
        model.eval()
        with torch.no_grad():
            output = model(node_features, adj, return_sequence=False)
        
        # GraphSAGE_BiGRU.forward returns spatial features: [batch, num_agents, hidden_features]
        # (BiGRU is only used in TemporalGraphSAGE_BiGRU for temporal processing)
        expected_shape = (self.batch_size, self.num_agents, self.graphsage_hidden_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any(), "GraphSAGE+BiGRU output contains NaN")
        
    def test_graphsage_bigru_sequence_output(self):
        """Test GraphSAGE+BiGRU with sequence output."""
        model = GraphSAGE_BiGRU(
            in_features=self.in_features,
            hidden_features=self.graphsage_hidden_dim,
            gru_hidden_size=self.gru_hidden_dim,
            dropout=0.0
        )
        
        node_features = torch.rand(self.batch_size, self.num_agents, self.in_features)
        adj = torch.eye(self.num_agents).unsqueeze(0).expand(self.batch_size, -1, -1)
        
        model.eval()
        with torch.no_grad():
            output_seq = model(node_features, adj, return_sequence=True)
        
        # Sequence output: [batch, num_agents, hidden_features]
        # Note: GraphSAGE_BiGRU only applies spatial aggregation in forward()
        expected_shape = (self.batch_size, self.num_agents, self.graphsage_hidden_dim)
        self.assertEqual(output_seq.shape, expected_shape)


class TestTemporalGraphSAGEBiGRU(unittest.TestCase):
    """Test Temporal GraphSAGE+BiGRU with observation history."""
    
    def setUp(self):
        """Set up temporal model parameters."""
        self.num_agents = 16
        self.in_features = 128
        self.graphsage_hidden_dim = 64
        self.gru_hidden_dim = 32
        self.history_length = 5
        self.batch_size = 4
        
    def test_temporal_graphsage_bigru_forward(self):
        """Test temporal GraphSAGE+BiGRU with observation history."""
        model = TemporalGraphSAGE_BiGRU(
            in_features=self.in_features,
            hidden_features=self.graphsage_hidden_dim,
            gru_hidden_size=self.gru_hidden_dim,
            history_length=self.history_length,
            dropout=0.0
        )
        
        # Input: [batch, history_length, num_agents, in_features]
        # Use contiguous tensor to avoid view issues
        temporal_features = torch.rand(
            self.batch_size, 
            self.history_length, 
            self.num_agents, 
            self.in_features
        ).contiguous()
        
        adj = torch.eye(self.num_agents)
        
        model.eval()
        with torch.no_grad():
            output = model(temporal_features, adj)
        
        # Output: [batch, graphsage_hidden_dim] (from output_dim property)
        expected_shape = (self.batch_size, self.graphsage_hidden_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any(), "Temporal GraphSAGE+BiGRU output contains NaN")


class TestMGMQEncoderEndToEnd(unittest.TestCase):
    """End-to-end test of MGMQ Encoder with SUMO-like inputs."""
    
    def setUp(self):
        """Set up MGMQ encoder and test parameters."""
        # SUMO observation format
        self.num_detectors = 12
        self.features_per_detector = 4
        self.obs_dim = self.num_detectors * self.features_per_detector  # 48
        
        # Model parameters
        self.num_agents = 16
        self.batch_size = 4
        self.gat_hidden_dim = 32
        self.gat_output_dim = 16
        self.gat_num_heads = 4
        self.graphsage_hidden_dim = 64
        self.gru_hidden_dim = 32
        
    def _create_encoder(self, window_size: int = 1) -> MGMQEncoder:
        """Create MGMQ encoder with given window size."""
        return MGMQEncoder(
            obs_dim=self.obs_dim * window_size,
            num_agents=self.num_agents,
            gat_hidden_dim=self.gat_hidden_dim,
            gat_output_dim=self.gat_output_dim,
            gat_num_heads=self.gat_num_heads,
            graphsage_hidden_dim=self.graphsage_hidden_dim,
            gru_hidden_dim=self.gru_hidden_dim,
            dropout=0.0,
            window_size=window_size
        )
        
    def test_encoder_forward_single_timestep(self):
        """Test encoder with single timestep observation."""
        encoder = self._create_encoder(window_size=1)
        
        # Create SUMO-like observation
        obs = torch.rand(self.batch_size, self.num_agents, self.obs_dim)
        obs = torch.clamp(obs, 0.0, 1.0)
        
        encoder.eval()
        with torch.no_grad():
            joint_emb, intersection_emb, network_emb = encoder(obs)
        
        # Verify shapes
        # Intersection embedding: gat_output_dim * gat_num_heads * 2 (coop + conf)
        expected_int_dim = self.gat_output_dim * self.gat_num_heads * 2
        self.assertEqual(intersection_emb.shape, (self.batch_size, expected_int_dim))
        
        # Network embedding: graphsage_hidden_dim (from MGMQEncoder.network_emb_dim)
        self.assertEqual(network_emb.shape, (self.batch_size, self.graphsage_hidden_dim))
        
        # Joint embedding: intersection_dim + network_dim
        self.assertEqual(
            joint_emb.shape, 
            (self.batch_size, expected_int_dim + self.graphsage_hidden_dim)
        )
        
        # Verify no NaN values
        self.assertFalse(torch.isnan(joint_emb).any(), "Joint embedding contains NaN")
        self.assertFalse(torch.isnan(intersection_emb).any(), "Intersection embedding contains NaN")
        self.assertFalse(torch.isnan(network_emb).any(), "Network embedding contains NaN")
        
    def test_encoder_forward_temporal(self):
        """Test encoder with temporal (history) observation."""
        window_size = 5
        encoder = self._create_encoder(window_size=window_size)
        
        # Create temporal SUMO-like observation
        # Flattened: [batch, num_agents, window_size * obs_dim]
        obs = torch.rand(self.batch_size, self.num_agents, self.obs_dim * window_size)
        obs = torch.clamp(obs, 0.0, 1.0)
        
        encoder.eval()
        with torch.no_grad():
            joint_emb, intersection_emb, network_emb = encoder(obs)
        
        # Verify no NaN values
        self.assertFalse(torch.isnan(joint_emb).any(), "Temporal joint embedding contains NaN")
        
    def test_encoder_with_single_agent(self):
        """Test encoder with single agent input."""
        encoder = MGMQEncoder(
            obs_dim=self.obs_dim,
            num_agents=1,
            gat_hidden_dim=self.gat_hidden_dim,
            gat_output_dim=self.gat_output_dim,
            gat_num_heads=self.gat_num_heads,
            graphsage_hidden_dim=self.graphsage_hidden_dim,
            gru_hidden_dim=self.gru_hidden_dim,
            dropout=0.0
        )
        
        # Single agent input: [batch, obs_dim]
        obs = torch.rand(self.batch_size, self.obs_dim)
        
        encoder.eval()
        with torch.no_grad():
            joint_emb, intersection_emb, network_emb = encoder(obs)
        
        self.assertFalse(torch.isnan(joint_emb).any(), "Single agent embedding contains NaN")
        
    def test_encoder_gradient_flow(self):
        """Test that gradients flow through the entire encoder."""
        encoder = self._create_encoder(window_size=1)
        encoder.train()
        
        obs = torch.rand(self.batch_size, self.num_agents, self.obs_dim, requires_grad=True)
        
        joint_emb, _, _ = encoder(obs)
        
        # Compute a simple loss and backpropagate
        loss = joint_emb.sum()
        loss.backward()
        
        # Check that gradients exist and are not NaN
        self.assertIsNotNone(obs.grad, "No gradient computed for input")
        self.assertFalse(torch.isnan(obs.grad).any(), "Input gradient contains NaN")
        
        # Check gradients in GAT layers
        for name, param in encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertFalse(
                    torch.isnan(param.grad).any(), 
                    f"Gradient for {name} contains NaN"
                )


class TestMGMQModelEndToEnd(unittest.TestCase):
    """End-to-end test of complete MGMQ Model with SUMO-like inputs."""
    
    def setUp(self):
        """Set up complete MGMQ model parameters."""
        self.obs_dim = 48  # 12 detectors * 4 features
        self.action_dim = 4  # 4 phases
        self.num_agents = 16
        self.batch_size = 4
        
    def test_model_forward_pass(self):
        """Test complete model forward pass with SUMO-like input."""
        model = MGMQModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            gat_hidden_dim=32,
            gat_output_dim=16,
            gat_num_heads=4,
            graphsage_hidden_dim=64,
            gru_hidden_dim=32,
            dropout=0.0
        )
        
        # Create SUMO-like observation
        obs = torch.rand(self.batch_size, self.num_agents, self.obs_dim)
        obs = torch.clamp(obs, 0.0, 1.0)
        
        model.eval()
        with torch.no_grad():
            action_mean, action_log_std, value = model(obs)
        
        # Verify output shapes
        self.assertEqual(action_mean.shape, (self.batch_size, self.action_dim))
        self.assertEqual(action_log_std.shape, (self.batch_size, self.action_dim))
        self.assertEqual(value.shape, (self.batch_size, 1))
        
        # Verify no NaN values
        self.assertFalse(torch.isnan(action_mean).any(), "Action mean contains NaN")
        self.assertFalse(torch.isnan(action_log_std).any(), "Action log_std contains NaN")
        self.assertFalse(torch.isnan(value).any(), "Value contains NaN")
        
    def test_model_action_sampling(self):
        """Test that model can sample valid actions."""
        model = MGMQModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents
        )
        
        obs = torch.rand(self.batch_size, self.num_agents, self.obs_dim)
        
        model.eval()
        with torch.no_grad():
            action_mean, action_log_std, _ = model(obs)
            
            # Sample actions using reparameterization trick
            std = torch.exp(action_log_std)
            eps = torch.randn_like(std)
            actions = action_mean + std * eps
        
        # Actions should be finite
        self.assertTrue(torch.isfinite(actions).all(), "Sampled actions are not finite")
        
    def test_model_training_step(self):
        """Test a complete training step with loss and backpropagation."""
        model = MGMQModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents
        )
        model.train()
        
        obs = torch.rand(self.batch_size, self.num_agents, self.obs_dim)
        
        # Forward pass
        action_mean, action_log_std, value = model(obs)
        
        # Dummy loss (policy loss + value loss)
        target_actions = torch.rand(self.batch_size, self.action_dim)
        target_values = torch.rand(self.batch_size, 1)
        
        policy_loss = ((action_mean - target_actions) ** 2).mean()
        value_loss = ((value - target_values) ** 2).mean()
        total_loss = policy_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
                self.assertFalse(
                    torch.isnan(param.grad).any(),
                    f"Gradient for {name} contains NaN"
                )
        
        self.assertGreater(grad_count, 0, "No gradients computed")


class TestAdjacencyMatrixConstruction(unittest.TestCase):
    """Test adjacency matrix construction for network topology."""
    
    def test_lane_cooperation_matrix(self):
        """Test lane cooperation matrix has correct structure."""
        adj_coop = get_lane_cooperation_matrix()
        
        # Should be 12x12 for standard intersection
        self.assertEqual(adj_coop.shape, (12, 12))
        
        # Should be symmetric
        self.assertTrue(
            torch.allclose(adj_coop, adj_coop.T),
            "Cooperation matrix should be symmetric"
        )
        
        # Diagonal should be 1 (self-loops)
        self.assertTrue(
            torch.allclose(adj_coop.diag(), torch.ones(12)),
            "Diagonal should be all ones"
        )
        
    def test_lane_conflict_matrix(self):
        """Test lane conflict matrix has correct structure."""
        adj_conf = get_lane_conflict_matrix()
        
        # Should be 12x12
        self.assertEqual(adj_conf.shape, (12, 12))
        
        # Should be symmetric
        self.assertTrue(
            torch.allclose(adj_conf, adj_conf.T),
            "Conflict matrix should be symmetric"
        )
        
    def test_cooperation_conflict_different(self):
        """Test that cooperation and conflict matrices are different."""
        adj_coop = get_lane_cooperation_matrix()
        adj_conf = get_lane_conflict_matrix()
        
        self.assertFalse(
            torch.allclose(adj_coop, adj_conf),
            "Cooperation and conflict matrices should be different"
        )


class TestDataFlowIntegration(unittest.TestCase):
    """Integration tests for data flow from SUMO observation to model output."""
    
    def test_complete_data_flow_multi_agent(self):
        """Test complete data flow for multi-agent scenario."""
        # Simulate SUMO environment output
        num_agents = 16
        obs_dim = 48
        batch_size = 2
        action_dim = 4
        
        # Step 1: Create SUMO-like observations (dict format)
        ts_ids = [f"ts_{i}" for i in range(num_agents)]
        obs_dict = {}
        for ts_id in ts_ids:
            # Each agent gets normalized observation
            obs_dict[ts_id] = np.clip(np.random.rand(obs_dim), 0, 1).astype(np.float32)
        
        # Step 2: Convert to tensor format for model
        obs_list = [obs_dict[ts_id] for ts_id in ts_ids]
        # Use clone().contiguous() to ensure tensor is properly laid out in memory
        obs_tensor = torch.tensor(np.stack(obs_list)).unsqueeze(0).clone().contiguous()  # [1, num_agents, obs_dim]
        obs_tensor = obs_tensor.expand(batch_size, -1, -1).clone().contiguous()  # [batch, num_agents, obs_dim]
        
        # Step 3: Create and run model
        model = MGMQModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents
        )
        model.eval()
        
        with torch.no_grad():
            action_mean, action_log_std, value = model(obs_tensor)
        
        # Step 4: Verify outputs
        self.assertEqual(action_mean.shape, (batch_size, action_dim))
        self.assertTrue(torch.isfinite(action_mean).all())
        self.assertTrue(torch.isfinite(value).all())
        
        # Step 5: Convert actions back to per-agent format if needed
        # (In actual training, each agent uses the same policy output)
        actions = action_mean.numpy()
        self.assertEqual(actions.shape, (batch_size, action_dim))
        
    def test_observation_normalization_robustness(self):
        """Test model handles edge cases in observation values."""
        model = MGMQModel(
            obs_dim=48,
            action_dim=4,
            num_agents=16
        )
        model.eval()
        
        # Test with all zeros
        obs_zeros = torch.zeros(2, 16, 48)
        with torch.no_grad():
            action_mean, _, value = model(obs_zeros)
        self.assertTrue(torch.isfinite(action_mean).all(), "Model fails with zero input")
        
        # Test with all ones
        obs_ones = torch.ones(2, 16, 48)
        with torch.no_grad():
            action_mean, _, value = model(obs_ones)
        self.assertTrue(torch.isfinite(action_mean).all(), "Model fails with ones input")
        
        # Test with extreme values (should be clipped by env, but test robustness)
        obs_extreme = torch.rand(2, 16, 48) * 2  # Some values > 1
        with torch.no_grad():
            action_mean, _, value = model(obs_extreme)
        self.assertTrue(torch.isfinite(action_mean).all(), "Model fails with extreme input")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
