
import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.mgmq_model import MGMQModel, MGMQEncoder

class TestMGMQModel(unittest.TestCase):
    def setUp(self):
        self.obs_dim = 48 # 12 lanes * 4 features
        self.action_dim = 4
        self.num_agents = 2
        self.batch_size = 5
        
    def test_encoder_shapes(self):
        """Test that encoder produces correct embedding shapes."""
        encoder = MGMQEncoder(
            obs_dim=self.obs_dim,
            num_agents=self.num_agents,
            gat_hidden_dim=32,
            gat_output_dim=16,
            gat_num_heads=2,
            graphsage_hidden_dim=32
        )
        
        # Input: [batch, num_agents, obs_dim]
        obs = torch.randn(self.batch_size, self.num_agents, self.obs_dim)
        
        joint_emb, intersection_emb, network_emb = encoder(obs)
        
        # Check shapes
        # Intersection emb: [batch, gat_output * heads * 2] (pooled over agents if agent_idx is None)
        expected_int_dim = 16 * 2 * 2 # 64
        self.assertEqual(intersection_emb.shape, (self.batch_size, expected_int_dim))
        
        # Network emb: [batch, graphsage_hidden] (pooled over agents)
        # Wait, code says: network_emb = network_emb_seq.mean(dim=1) -> [batch, hidden]
        self.assertEqual(network_emb.shape, (self.batch_size, 32))
        
        # Joint emb: [batch, int_dim + net_dim] (pooled intersection if no agent_idx)
        # If agent_idx is None, it uses mean of intersection embeddings
        expected_joint_dim = expected_int_dim + 32
        self.assertEqual(joint_emb.shape, (self.batch_size, expected_joint_dim))

    def test_model_forward(self):
        """Test full model forward pass."""
        model = MGMQModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents
        )
        
        obs = torch.randn(self.batch_size, self.num_agents, self.obs_dim)
        
        action_mean, action_log_std, value = model(obs)
        
        # Check output shapes
        self.assertEqual(action_mean.shape, (self.batch_size, self.action_dim))
        self.assertEqual(action_log_std.shape, (self.batch_size, self.action_dim))
        self.assertEqual(value.shape, (self.batch_size, 1))

    def test_single_agent_shape(self):
        """Test model with single agent input."""
        model = MGMQModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_agents=1
        )
        
        # Input: [batch, obs_dim] (no agent dim)
        obs = torch.randn(self.batch_size, self.obs_dim)
        
        action_mean, _, _ = model(obs)
        self.assertEqual(action_mean.shape, (self.batch_size, self.action_dim))

if __name__ == '__main__':
    unittest.main()
