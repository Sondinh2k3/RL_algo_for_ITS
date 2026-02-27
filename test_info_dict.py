import json
from src.environment.rllib_utils import SumoMultiAgentEnv
import numpy as np

# Create Env
with open('src/config/intersection_config.json') as f:
    ts_ids = [ts["id"] for ts in json.load(f)["intersections"] if ts.get("is_agent", True)]

env_config = {
    "net_file": "network/grid4x4/grid4x4.net.xml",
    "route_file": "network/grid4x4/grid4x4.rou.xml,network/grid4x4/grid4x4-demo.rou.xml",
    "num_seconds": 200,
    "use_gui": False,
    "ts_ids": ts_ids,
    "reward_fn": "diff-waiting-time",
    "normalize_reward": True
}

env = SumoMultiAgentEnv(**env_config)
obs, info = env.reset(seed=42)

actions = {ts: 0 for ts in ts_ids}
obs, rewards, term, trunc, info = env.step(actions)

print("--- ENV RETURNED INFO ---")
print(f"Keys in info: {list(info.keys())}")
sample_ts = ts_ids[0]
print(f"Info for {sample_ts}: {info.get(sample_ts)}")

print("\n--- SIMULATING RLLIB PROCESSING ---")
# RLlib MultiAgentEpisode does this (roughly):
# It maps info to agents
agents = list(rewards.keys())
print(f"Are agent keys dicts? {isinstance(info.get(sample_ts), dict)}")
print(f"Does it have raw_reward? {'raw_reward' in info.get(sample_ts, {})}")

env.close()
