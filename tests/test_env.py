import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from environment.drl_algo.env import SumoEnvironment

# Mock SumoSimulator để không cần chạy SUMO thật
class MockSimulator:
    def __init__(self, *args, **kwargs):
        self.sumo_seed = None
        self.closed = False
        self.ts_ids = ["ts_1", "ts_2"]

    def initialize(self):
        # Trả về trạng thái ban đầu cho 2 agent
        return {"ts_1": np.array([0, 0]), "ts_2": np.array([0, 0])}

    def reset(self):
        # Trả về trạng thái reset cho 2 agent
        return {"ts_1": np.array([0, 0]), "ts_2": np.array([0, 0])}

    def step(self, actions):
        # Trả về kết quả step cho 2 agent
        obs = {"ts_1": np.array([1, 2]), "ts_2": np.array([3, 4])}
        rewards = {"ts_1": 1.0, "ts_2": 2.0}
        dones = {"ts_1": False, "ts_2": False, "__all__": False}
        info = {"step": 1}
        return obs, rewards, dones, info

    def get_agent_ids(self):
        return ["ts_1", "ts_2"]

    def close(self):
        self.closed = True

    def get_sim_step(self):
        return 1

    def get_metrics(self):
        return {"metric": 42}

    def get_system_info(self):
        return {"system_total_running": 2}

    def get_per_agent_info(self):
        return {"ts_1_stopped": 0, "ts_2_stopped": 1}

    def get_rgb_array(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

# Fixture khởi tạo môi trường với mock simulator
@pytest.fixture
def env():
    """Create environment with mocked simulator to avoid SUMO initialization."""
    with patch('environment.drl_algo.env.SumoSimulator', return_value=MockSimulator()):
        env = SumoEnvironment(
            net_file="net.xml",
            route_file="route.xml",
            out_csv_name=None,
            use_gui=False,
            virtual_display=(800, 600),
            begin_time=0,
            num_seconds=100,
            ts_ids=["ts_1", "ts_2"],
            single_agent=False,
        )
    return env

def test_init(env):
    assert env.ts_ids == ["ts_1", "ts_2"]
    assert env.simulator is not None

def test_reset(env):
    obs = env.reset()
    assert isinstance(obs, dict)
    assert "ts_1" in obs and "ts_2" in obs

def test_step(env):
    actions = {"ts_1": 0, "ts_2": 1}
    obs, rewards, dones, info = env.step(actions)
    assert isinstance(obs, dict)
    assert isinstance(rewards, dict)
    assert isinstance(dones, dict)
    assert isinstance(info, dict)
    assert "ts_1" in obs and "ts_2" in obs
    assert "step" in info

def test_compute_info(env):
    info = env._compute_info()
    assert isinstance(info, dict)
    assert "step" in info
    assert "metric" in info
    assert "system_total_running" in info
    assert "ts_1_stopped" in info
    assert "ts_2_stopped" in info

def test_close(env):
    env.close()
    assert env.simulator.closed is True

def test_render_rgb_array(env):
    env.render_mode = "rgb_array"
    arr = env.render()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (100, 100, 3)

def test_save_csv(env, tmp_path):
    # Test lưu file csv (không lỗi, file được tạo)
    env.metrics = [{"step": 1, "metric": 42}]
    out_csv = tmp_path / "test.csv"
    env.save_csv(str(out_csv), 1)
    assert out_csv.with_name(out_csv.name + f"_conn{env.label}_ep1.csv").exists()

def test_single_agent_reset_and_step():
    """Test single-agent mode."""
    with patch('environment.drl_algo.env.SumoSimulator', return_value=MockSimulator()):
        env = SumoEnvironment(
            net_file="net.xml",
            route_file="route.xml",
            out_csv_name=None,
            use_gui=False,
            virtual_display=(800, 600),
            begin_time=0,
            num_seconds=100,
            ts_ids=["ts_1"],
            single_agent=True,
        )
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    actions = 0
    obs, reward, terminated, truncated, info = env.step(actions)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)