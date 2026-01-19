"""Unit tests for LocalTemporalMGMQEncoder.

Tests the Local Temporal MGMQ Encoder designed for Pre-packaged Observations.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestLocalTemporalMGMQEncoder:
    """Tests for LocalTemporalMGMQEncoder class."""
    
    @pytest.fixture
    def encoder_config(self):
        """Default encoder configuration."""
        return {
            "obs_dim": 48,
            "max_neighbors": 4,
            "window_size": 5,
            "gat_hidden_dim": 64,
            "gat_output_dim": 32,
            "gat_num_heads": 4,
            "graphsage_hidden_dim": 64,
            "gru_hidden_dim": 32,
            "dropout": 0.0,  # No dropout for deterministic tests
        }
    
    @pytest.fixture
    def mock_dict_observation(self, encoder_config):
        """Create a mock Dict observation for testing."""
        B = 4  # batch size
        T = encoder_config["window_size"]
        K = encoder_config["max_neighbors"]
        feature_dim = encoder_config["obs_dim"]
        
        return {
            "self_features": torch.rand(B, T, feature_dim),
            "neighbor_features": torch.rand(B, K, T, feature_dim),
            "neighbor_mask": torch.tensor([
                [1.0, 1.0, 0.0, 0.0],  # 2 valid neighbors
                [1.0, 1.0, 1.0, 0.0],  # 3 valid neighbors
                [1.0, 0.0, 0.0, 0.0],  # 1 valid neighbor
                [1.0, 1.0, 1.0, 1.0],  # 4 valid neighbors
            ]),
        }
    
    def test_encoder_instantiation(self, encoder_config):
        """Test that encoder can be instantiated with default config."""
        from src.models.mgmq_model import LocalTemporalMGMQEncoder
        
        encoder = LocalTemporalMGMQEncoder(**encoder_config)
        
        assert encoder is not None
        assert encoder.obs_dim == encoder_config["obs_dim"]
        assert encoder.max_neighbors == encoder_config["max_neighbors"]
        assert encoder.window_size == encoder_config["window_size"]
    
    def test_encoder_output_dim(self, encoder_config):
        """Test that output_dim matches expected joint embedding dimension."""
        from src.models.mgmq_model import LocalTemporalMGMQEncoder
        
        encoder = LocalTemporalMGMQEncoder(**encoder_config)
        
        # Joint dim = intersection_emb_dim + network_emb_dim
        expected_intersection_dim = (encoder_config["gat_output_dim"] * encoder_config["gat_num_heads"]) * 2
        expected_network_dim = encoder_config["graphsage_hidden_dim"]
        expected_joint_dim = expected_intersection_dim + expected_network_dim
        
        assert encoder.output_dim == expected_joint_dim
    
    def test_forward_output_shape(self, encoder_config, mock_dict_observation):
        """Test that forward pass produces correct output shape."""
        from src.models.mgmq_model import LocalTemporalMGMQEncoder
        
        encoder = LocalTemporalMGMQEncoder(**encoder_config)
        
        output = encoder(mock_dict_observation)
        
        B = mock_dict_observation["self_features"].size(0)
        expected_shape = (B, encoder.output_dim)
        
        assert output.shape == expected_shape
    
    def test_gradient_flow(self, encoder_config, mock_dict_observation):
        """Test that gradients flow through the encoder."""
        from src.models.mgmq_model import LocalTemporalMGMQEncoder
        
        encoder = LocalTemporalMGMQEncoder(**encoder_config)
        
        # Require grad for input
        mock_dict_observation["self_features"].requires_grad_(True)
        
        output = encoder(mock_dict_observation)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert mock_dict_observation["self_features"].grad is not None
        assert mock_dict_observation["self_features"].grad.abs().sum() > 0
    
    def test_star_adjacency_masking(self, encoder_config):
        """Test that _build_star_adjacency correctly applies mask."""
        from src.models.mgmq_model import LocalTemporalMGMQEncoder
        
        encoder = LocalTemporalMGMQEncoder(**encoder_config)
        
        # Create test mask: [batch, K]
        mask = torch.tensor([
            [1.0, 1.0, 0.0, 0.0],  # 2 valid neighbors
            [1.0, 0.0, 0.0, 0.0],  # 1 valid neighbor
        ])
        
        adj = encoder._build_star_adjacency(mask)
        
        # Check shape: [B, 1+K, 1+K]
        assert adj.shape == (2, 5, 5)
        
        # Check self-loop always present
        assert adj[0, 0, 0] == 1.0
        assert adj[1, 0, 0] == 1.0
        
        # Check neighbor connections match mask
        # Sample 0: neighbors 0, 1 are valid
        assert adj[0, 0, 1] == 1.0  # self -> neighbor 0
        assert adj[0, 0, 2] == 1.0  # self -> neighbor 1
        assert adj[0, 0, 3] == 0.0  # neighbor 2 is padded
        assert adj[0, 0, 4] == 0.0  # neighbor 3 is padded
        
        # Sample 1: only neighbor 0 is valid
        assert adj[1, 0, 1] == 1.0
        assert adj[1, 0, 2] == 0.0
    
    def test_no_valid_neighbors(self, encoder_config):
        """Test encoder works with no valid neighbors (all padding)."""
        from src.models.mgmq_model import LocalTemporalMGMQEncoder
        
        encoder = LocalTemporalMGMQEncoder(**encoder_config)
        
        B, T, K = 2, encoder_config["window_size"], encoder_config["max_neighbors"]
        feature_dim = encoder_config["obs_dim"]
        
        obs_dict = {
            "self_features": torch.rand(B, T, feature_dim),
            "neighbor_features": torch.zeros(B, K, T, feature_dim),  # All zeros
            "neighbor_mask": torch.zeros(B, K),  # No valid neighbors
        }
        
        # Should not crash
        output = encoder(obs_dict)
        
        assert output.shape == (B, encoder.output_dim)
        assert not torch.isnan(output).any()


class TestLocalTemporalMGMQTorchModel:
    """Tests for LocalTemporalMGMQTorchModel RLlib wrapper."""
    
    @pytest.fixture
    def model_config(self):
        """Model config for RLlib."""
        return {
            "custom_model_config": {
                "obs_dim": 48,
                "max_neighbors": 4,
                "window_size": 5,
                "gat_hidden_dim": 64,
                "gat_output_dim": 32,
                "gat_num_heads": 4,
                "graphsage_hidden_dim": 64,
                "gru_hidden_dim": 32,
                "policy_hidden_dims": [64],
                "value_hidden_dims": [64],
                "dropout": 0.0,
            }
        }
    
    @pytest.fixture
    def mock_obs_space(self):
        """Mock Dict observation space."""
        from gymnasium import spaces
        import numpy as np
        
        T = 5
        K = 4
        feature_dim = 48
        
        return spaces.Dict({
            "self_features": spaces.Box(0, 1, shape=(T, feature_dim), dtype=np.float32),
            "neighbor_features": spaces.Box(0, 1, shape=(K, T, feature_dim), dtype=np.float32),
            "neighbor_mask": spaces.Box(0, 1, shape=(K,), dtype=np.float32),
        })
    
    @pytest.fixture
    def mock_action_space(self):
        """Mock continuous action space."""
        from gymnasium import spaces
        import numpy as np
        
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    
    def test_model_instantiation(self, mock_obs_space, mock_action_space, model_config):
        """Test that model can be instantiated."""
        from src.models.mgmq_model import LocalTemporalMGMQTorchModel
        
        model = LocalTemporalMGMQTorchModel(
            obs_space=mock_obs_space,
            action_space=mock_action_space,
            num_outputs=8,  # 4 means + 4 log stds
            model_config=model_config,
            name="test_local_mgmq"
        )
        
        assert model is not None
        assert model.mgmq_encoder is not None
    
    def test_forward_pass(self, mock_obs_space, mock_action_space, model_config):
        """Test forward pass with Dict observation."""
        from src.models.mgmq_model import LocalTemporalMGMQTorchModel
        
        model = LocalTemporalMGMQTorchModel(
            obs_space=mock_obs_space,
            action_space=mock_action_space,
            num_outputs=8,
            model_config=model_config,
            name="test_local_mgmq"
        )
        
        B = 4
        T = 5
        K = 4
        feature_dim = 48
        
        input_dict = {
            "obs": {
                "self_features": torch.rand(B, T, feature_dim),
                "neighbor_features": torch.rand(B, K, T, feature_dim),
                "neighbor_mask": torch.ones(B, K),
            }
        }
        
        policy_out, state = model.forward(input_dict, [], None)
        
        assert policy_out.shape == (B, 8)
        assert state == []
    
    def test_value_function(self, mock_obs_space, mock_action_space, model_config):
        """Test value function after forward pass."""
        from src.models.mgmq_model import LocalTemporalMGMQTorchModel
        
        model = LocalTemporalMGMQTorchModel(
            obs_space=mock_obs_space,
            action_space=mock_action_space,
            num_outputs=8,
            model_config=model_config,
            name="test_local_mgmq"
        )
        
        B = 4
        input_dict = {
            "obs": {
                "self_features": torch.rand(B, 5, 48),
                "neighbor_features": torch.rand(B, 4, 5, 48),
                "neighbor_mask": torch.ones(B, 4),
            }
        }
        
        # Must call forward first
        model.forward(input_dict, [], None)
        
        value = model.value_function()
        
        assert value.shape == (B,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
