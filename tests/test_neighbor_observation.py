"""Unit tests for NeighborTemporalObservationFunction.

This tests the pre-packaged observation with neighbor features for Local GNN.
"""

import pytest
import numpy as np
from gymnasium import spaces
from unittest.mock import Mock, MagicMock


class TestNeighborTemporalObservationFunction:
    """Tests for NeighborTemporalObservationFunction class."""
    
    @pytest.fixture
    def mock_traffic_signal(self):
        """Create a mock TrafficSignal object."""
        ts = Mock()
        ts.id = "tl_1"
        ts.detectors_e2 = [f"e2_{i}" for i in range(12)]  # 12 detectors
        ts.window_size = 5  # History length
        
        # Mock get_observation_history - returns list of T observations
        ts.get_observation_history = Mock(return_value=[
            np.random.random(48).astype(np.float32) for _ in range(5)
        ])
        
        # Mock detector methods
        ts.get_lanes_density_by_detectors = Mock(return_value=[0.5] * 12)
        ts.get_lanes_queue_by_detectors = Mock(return_value=[0.3] * 12)
        ts.get_lanes_occupancy_by_detectors = Mock(return_value=[0.4] * 12)
        ts.get_lanes_average_speed_by_detectors = Mock(return_value=[0.7] * 12)
        
        return ts
    
    @pytest.fixture
    def mock_neighbor_provider(self):
        """Create a mock NeighborProvider."""
        provider = Mock()
        provider.get_neighbor_ids = Mock(return_value=["tl_2", "tl_3", "tl_4"])
        
        # Return mock observation history for each neighbor
        def get_obs_history(ts_id, window_size):
            if ts_id in ["tl_2", "tl_3", "tl_4"]:
                return [np.random.random(48).astype(np.float32) for _ in range(window_size)]
            return None
        
        provider.get_observation_history = Mock(side_effect=get_obs_history)
        return provider
    
    def test_observation_space_shape(self, mock_traffic_signal, mock_neighbor_provider):
        """Test that observation space has correct Dict structure and shapes."""
        from src.environment.drl_algo.observations import NeighborTemporalObservationFunction
        
        obs_fn = NeighborTemporalObservationFunction(
            ts=mock_traffic_signal,
            neighbor_provider=mock_neighbor_provider,
            max_neighbors=4,
            window_size=5
        )
        
        obs_space = obs_fn.observation_space()
        
        # Check it's a Dict space
        assert isinstance(obs_space, spaces.Dict)
        
        # Check required keys exist
        assert "self_features" in obs_space.spaces
        assert "neighbor_features" in obs_space.spaces
        assert "neighbor_mask" in obs_space.spaces
        
        # Check shapes
        T = 5  # window_size
        K = 4  # max_neighbors
        feature_dim = 48  # 4 features * 12 detectors
        
        assert obs_space["self_features"].shape == (T, feature_dim)
        assert obs_space["neighbor_features"].shape == (K, T, feature_dim)
        assert obs_space["neighbor_mask"].shape == (K,)
    
    def test_observation_call_returns_dict(self, mock_traffic_signal, mock_neighbor_provider):
        """Test that __call__ returns a dict with correct keys."""
        from src.environment.drl_algo.observations import NeighborTemporalObservationFunction
        
        obs_fn = NeighborTemporalObservationFunction(
            ts=mock_traffic_signal,
            neighbor_provider=mock_neighbor_provider,
            max_neighbors=4,
            window_size=5
        )
        
        obs = obs_fn()
        
        # Check it's a dict with correct keys
        assert isinstance(obs, dict)
        assert "self_features" in obs
        assert "neighbor_features" in obs
        assert "neighbor_mask" in obs
    
    def test_observation_shapes(self, mock_traffic_signal, mock_neighbor_provider):
        """Test that returned observation arrays have correct shapes."""
        from src.environment.drl_algo.observations import NeighborTemporalObservationFunction
        
        obs_fn = NeighborTemporalObservationFunction(
            ts=mock_traffic_signal,
            neighbor_provider=mock_neighbor_provider,
            max_neighbors=4,
            window_size=5
        )
        
        obs = obs_fn()
        
        T = 5
        K = 4
        feature_dim = 48
        
        assert obs["self_features"].shape == (T, feature_dim)
        assert obs["neighbor_features"].shape == (K, T, feature_dim)
        assert obs["neighbor_mask"].shape == (K,)
    
    def test_neighbor_mask_values(self, mock_traffic_signal, mock_neighbor_provider):
        """Test that neighbor mask correctly indicates valid neighbors."""
        from src.environment.drl_algo.observations import NeighborTemporalObservationFunction
        
        obs_fn = NeighborTemporalObservationFunction(
            ts=mock_traffic_signal,
            neighbor_provider=mock_neighbor_provider,
            max_neighbors=4,
            window_size=5
        )
        
        obs = obs_fn()
        
        # We have 3 neighbors (tl_2, tl_3, tl_4), so first 3 should be 1.0
        # and last one should be 0.0 (padding)
        assert obs["neighbor_mask"][0] == 1.0
        assert obs["neighbor_mask"][1] == 1.0
        assert obs["neighbor_mask"][2] == 1.0
        assert obs["neighbor_mask"][3] == 0.0  # Padding
    
    def test_observation_values_clipped(self, mock_traffic_signal, mock_neighbor_provider):
        """Test that all observation values are clipped to [0, 1]."""
        from src.environment.drl_algo.observations import NeighborTemporalObservationFunction
        
        obs_fn = NeighborTemporalObservationFunction(
            ts=mock_traffic_signal,
            neighbor_provider=mock_neighbor_provider,
            max_neighbors=4,
            window_size=5
        )
        
        obs = obs_fn()
        
        # Check self_features clipped
        assert obs["self_features"].min() >= 0.0
        assert obs["self_features"].max() <= 1.0
        
        # Check neighbor_features clipped
        assert obs["neighbor_features"].min() >= 0.0
        assert obs["neighbor_features"].max() <= 1.0
    
    def test_no_neighbor_provider(self, mock_traffic_signal):
        """Test behavior when neighbor_provider is None."""
        from src.environment.drl_algo.observations import NeighborTemporalObservationFunction
        
        obs_fn = NeighborTemporalObservationFunction(
            ts=mock_traffic_signal,
            neighbor_provider=None,  # No provider
            max_neighbors=4,
            window_size=5
        )
        
        obs = obs_fn()
        
        # neighbor_features should be zeros
        assert np.allclose(obs["neighbor_features"], 0.0)
        
        # mask should be all zeros (no valid neighbors)
        assert np.allclose(obs["neighbor_mask"], 0.0)


class TestNeighborProvider:
    """Tests for NeighborProvider class."""
    
    def test_get_neighbor_ids(self):
        """Test that get_neighbor_ids returns correct neighbors."""
        from src.sim.Sumo_sim import NeighborProvider
        
        adjacency_map = {
            "tl_1": ["tl_2", "tl_3"],
            "tl_2": ["tl_1", "tl_4"],
        }
        
        provider = NeighborProvider(
            traffic_signals={},
            adjacency_map=adjacency_map,
            max_neighbors=4
        )
        
        neighbors = provider.get_neighbor_ids("tl_1")
        assert neighbors == ["tl_2", "tl_3"]
        
        neighbors = provider.get_neighbor_ids("tl_2")
        assert neighbors == ["tl_1", "tl_4"]
        
        # Unknown ts_id should return empty list
        neighbors = provider.get_neighbor_ids("unknown")
        assert neighbors == []
    
    def test_max_neighbors_limit(self):
        """Test that get_neighbor_ids respects max_neighbors."""
        from src.sim.Sumo_sim import NeighborProvider
        
        adjacency_map = {
            "tl_1": ["tl_2", "tl_3", "tl_4", "tl_5", "tl_6"],  # 5 neighbors
        }
        
        provider = NeighborProvider(
            traffic_signals={},
            adjacency_map=adjacency_map,
            max_neighbors=3  # Limit to 3
        )
        
        neighbors = provider.get_neighbor_ids("tl_1")
        assert len(neighbors) == 3
        assert neighbors == ["tl_2", "tl_3", "tl_4"]


class TestBuildAdjacencyMapFromNetwork:
    """Tests for build_adjacency_map_from_network function."""
    
    def test_empty_ts_ids(self, tmp_path):
        """Test with empty ts_ids list."""
        from src.sim.Sumo_sim import build_adjacency_map_from_network
        
        # Create a minimal net.xml file
        net_file = tmp_path / "test.net.xml"
        net_file.write_text("""<?xml version="1.0"?>
<net>
    <edge id="e1" from="j1" to="j2"/>
</net>
""")
        
        adjacency = build_adjacency_map_from_network(str(net_file), [])
        assert adjacency == {}
    
    def test_nonexistent_file(self):
        """Test with nonexistent network file."""
        from src.sim.Sumo_sim import build_adjacency_map_from_network
        
        adjacency = build_adjacency_map_from_network("/nonexistent/file.net.xml", ["tl_1"])
        assert adjacency == {"tl_1": []}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
