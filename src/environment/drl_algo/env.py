"""Environment for Traffic Signal Control using SimulatorAPI."""

import os
import sys
import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

try:
    # pettingzoo 1.25+
    from pettingzoo.utils import AgentSelector
except ImportError:
    # pettingzoo 1.24 or earlier
    from pettingzoo.utils import agent_selector as AgentSelector

from pettingzoo.utils.conversions import parallel_wrapper_fn

from .observations import DefaultObservationFunction, ObservationFunction
from .traffic_signal import TrafficSignal


# Add parent directories to path for imports
_src_path = Path(__file__).parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from sim.simulator_api import SimulatorAPI
from sim.Sumo_sim import SumoSimulator 



def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class SumoEnvironment(gym.Env):
    """SUMO Environment for Traffic Signal Control.

    Class that implements a gym.Env interface for traffic signal control using the SUMO simulator.
    See https://sumo.dlr.de/docs/ for details on SUMO.
    See https://gymnasium.farama.org/ for details on gymnasium.

    Args:
        net_file (str): SUMO .net.xml file
        route_file (str): SUMO .rou.xml file
        out_csv_name (Optional[str]): name of the .csv output with simulation results. If None, no output is generated
        use_gui (bool): Whether to run SUMO simulation with the SUMO GUI
        virtual_display (Optional[Tuple[int,int]]): Resolution of the virtual display for rendering
        begin_time (int): The time step (in seconds) the simulation starts. Default: 0
        num_seconds (int): Number of simulated seconds on SUMO. The duration in seconds of the simulation. Default: 20000
        max_depart_delay (int): Vehicles are discarded if they could not be inserted after max_depart_delay seconds. Default: -1 (no delay)
        waiting_time_memory (int): Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime). Default: 1000
        time_to_teleport (int): Time in seconds to teleport a vehicle to the end of the edge if it is stuck. Default: -1 (no teleport)
        delta_time (int): DEPRECATED - Now always equals cycle_time. Simulation seconds between actions.
        yellow_time (int): Duration of the yellow phase. Default: 2 seconds
        min_green (int): Minimum green time in a phase. Default: 5 seconds
        max_green (int): Max green time in a phase. Default: 60 seconds. Warning: This parameter is currently ignored!
        enforce_max_green (bool): If true, it enforces the max green time and selects the next green phase when the max green time is reached. Default: False
        single_agent (bool): If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (returns dict of observations, rewards, dones, infos).
        reward_fn (str/function/dict/List): String with the name of the reward function used by the agents, a reward function, dictionary with reward functions assigned to individual traffic lights by their keys, or a List of reward functions.
        reward_weights (List[float]/np.ndarray): Weights for linearly combining the reward functions, in case reward_fn is a list. If it is None, the reward returned will be a np.ndarray. Default: None
        observation_class (ObservationFunction): Inherited class which has both the observation function and observation space.
        add_system_info (bool): If true, it computes system metrics (total queue, total waiting time, average speed) in the info dictionary.
        add_per_agent_info (bool): If true, it computes per-agent (per-traffic signal) metrics (average accumulated waiting time, average queue) in the info dictionary.
        sumo_seed (int/string): Random seed for sumo. If 'random' it uses a randomly chosen seed.
        ts_ids (Optional[List[str]]): List of traffic light IDs to be controlled by SUMO-RL. If None, all traffic lights in the simulation are controlled.
        fixed_ts (bool): If true, it will follow the phase configuration in the route_file and ignore the actions given in the :meth:`step` method.
        sumo_warnings (bool): If true, it will print SUMO warnings.
        additional_sumo_cmd (str): Additional SUMO command line arguments.
        render_mode (str): Mode of rendering. Can be 'human' or 'rgb_array'. Default: None
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self,
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        cycle_time: int = 90,
        delta_time: int = None,  # DEPRECATED: Now always equals cycle_time
        yellow_time: int = 3,
        min_green: int = 5,
        max_green: int = 50,
        enforce_max_green: bool = False,
        single_agent: bool = False,
        reward_fn: Union[str, Callable, dict, List] = "diff-waiting-time",
        reward_weights: Optional[List[float]] = None,
        observation_class: type[ObservationFunction] = DefaultObservationFunction,
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        ts_ids: Optional[List[str]] = None,
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
        preprocessing_config: Optional[str] = None,
        window_size: int = 1,  # Added for Spatio-Temporal
        use_phase_standardizer: bool = False,  # Enable phase standardization for action translation
        use_neighbor_obs: bool = False,  # Enable neighbor observation for Local GNN
        max_neighbors: int = 4,  # Maximum neighbors (K) for neighbor observation
    ) -> None:
        """Initialize the environment."""
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui

        assert cycle_time > yellow_time, "Cycle time must be at least greater than yellow time."
        assert cycle_time > min_green, "Cycle time must be at least greater than min green time."
        assert max_green > min_green, "Max green time must be greater than min green time."
        
        # IMPORTANT: delta_time (time between actions) MUST equal cycle_time
        # This ensures agent makes exactly one decision per traffic light cycle
        if delta_time is not None and delta_time != cycle_time:
            print(f"Warning: delta_time ({delta_time}) is ignored. Using cycle_time ({cycle_time}) as time between actions.")
        delta_time = cycle_time  # Force delta_time = cycle_time

        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.delta_time = cycle_time  # seconds on sumo at each step (MUST equal cycle_time)
        self.cycle_time = cycle_time
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.preprocessing_config = preprocessing_config
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        self.window_size = window_size
        SumoEnvironment.CONNECTION_LABEL += 1
        
        # Create simulator instance
        self.simulator = SumoSimulator(
            net_file=self._net,
            route_file=self._route,
            label=self.label,
            use_gui=self.use_gui,
            virtual_display=self.virtual_display,
            begin_time=self.begin_time,
            delta_time=self.cycle_time,  # Use cycle_time as delta_time for TrafficSignal
            yellow_time=self.yellow_time,
            min_green=self.min_green,
            max_green=self.max_green,
            enforce_max_green=self.enforce_max_green,
            sumo_seed=str(self.sumo_seed) if self.sumo_seed != "random" else "random",
            additional_sumo_cmd=self.additional_sumo_cmd.split() if self.additional_sumo_cmd else [],
            sumo_warnings=self.sumo_warnings,
            max_depart_delay=self.max_depart_delay,
            waiting_time_memory=self.waiting_time_memory,
            time_to_teleport=self.time_to_teleport,
            preprocessing_config=self.preprocessing_config,
            window_size=self.window_size,  # Pass window_size to Simulator
            num_seconds=num_seconds,
            reward_fn=self.reward_fn,  # Pass reward function to Simulator
            reward_weights=self.reward_weights,  # Pass reward weights to Simulator
            use_phase_standardizer=use_phase_standardizer,  # Pass phase standardizer flag
            use_neighbor_obs=use_neighbor_obs,  # Pass neighbor observation flag
            max_neighbors=max_neighbors,  # Pass max neighbors
            fixed_ts=self.fixed_ts, # Pass fixed_ts flag to Simulator
        )
        
        # Initialize simulator and get initial state
        initial_state = self.simulator.initialize()
        
        # Get traffic signal IDs from simulator
        # IMPORTANT: Only include signals that have valid TrafficSignal objects
        # (those with E2 detectors). Other signals were skipped during initialization.
        if ts_ids is None:
            # Get only the valid traffic signal IDs (those that have TrafficSignal objects)
            valid_ts_ids = list(self.simulator.traffic_signals.keys())
            if not valid_ts_ids:
                raise RuntimeError(
                    "[SumoEnvironment] No valid traffic signals found! "
                    "Check that E2 detectors are configured in the network."
                )
            self.ts_ids = valid_ts_ids
        else:
            # Filter provided ts_ids to only include valid ones
            valid_ts_ids = [ts for ts in ts_ids if ts in self.simulator.traffic_signals]
            if not valid_ts_ids:
                raise RuntimeError(
                    f"[SumoEnvironment] None of provided ts_ids {ts_ids} have valid TrafficSignal objects!"
                )
            if len(valid_ts_ids) < len(ts_ids):
                skipped = [ts for ts in ts_ids if ts not in valid_ts_ids]
                print(f"[SumoEnv] Warning: Skipped {len(skipped)} invalid ts_ids: {skipped}")
            self.ts_ids = valid_ts_ids
            
        self.observation_class = observation_class

        self.vehicles = dict()
        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}


        # Debug logging is disabled by default for performance
        # Uncomment to enable: self.enable_debug_logging(True, level=2)
    # def _setup_reward_functions(self):
    #     """Setup reward functions for each traffic signal."""
    #     if not isinstance(self.reward_fn, dict):
    #         self.reward_fn = {ts: self.reward_fn for ts in self.ts_ids}

    # def _start_simulation(self):
    #     """Start a new simulation episode."""
    #     self.simulator.reset()  # Reset simulator to initial state

    # Đặt lại môi trường về trạng thái ban đầu khi bắt đầu một episode mới.
    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)

        # Lần đầu khởi tạo sẽ không cần phải đóng môi trường và lưu kết quả
        if self.episode != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []

        # Reset simulator with new seed if provided
        if seed is not None:
            self.sumo_seed = seed
            
        # Reset simulation and get initial observations
        observations = self.simulator.reset(seed=seed)
        self.observations = observations
        
        # Reset metrics
        self.vehicles = dict()
        self.metrics = []
        
        if self.single_agent:
            return observations[self.ts_ids[0]], self._compute_info()
        else:
            return observations, self._compute_info()

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.simulator.get_sim_step()

    # Thực hiện một bước trong môi trường với hành động đã cho.
    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        # Convert single agent action to dict if needed
        if self.single_agent:
            actions = {self.ts_ids[0]: action}
        else:
            actions = action

        # No action in fixed timing mode => in fixed time, don't action
        if self.fixed_ts:
            actions = {}

        # Step simulator and get results
        observations, rewards, dones, info = self.simulator.step(actions)
        
        # Update internal state
        self.observations = observations
        self.rewards = rewards
        
        # Add metrics to info
        info.update(self._compute_info())
        
        # Episode ends when sim_step >= max_steps
        terminated = False  # no terminal states in this environment - env just end when timeout or end of vehicles
        truncated = dones["__all__"]

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info

    def _compute_info(self):
        """Compute additional information about the environment state."""
        info = {"step": self.simulator.get_sim_step()}
        
        # Get standard metrics from simulator
        metrics = self.simulator.get_metrics()
        info.update(metrics)
        
        # Add custom metrics if needed
        if self.add_system_info:
            info.update(self.simulator.get_system_info())
        if self.add_per_agent_info:
            info.update(self.simulator.get_per_agent_info())
            
        self.metrics.append(info.copy())
        return info

    @property
    def observation_space(self):
        """Return the observation space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.simulator.get_observation_space(self.ts_ids[0])

    @property
    def action_space(self):
        """Return the action space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.simulator.get_action_space(self.ts_ids[0])

    def observation_spaces(self, ts_id: str):
        """Return the observation space of a traffic signal."""
        return self.simulator.get_observation_space(ts_id)

    def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
        """Return the action space of a traffic signal."""
        return self.simulator.get_action_space(ts_id)

    def close(self):
        """Close the environment and stop the simulation."""
        if self.simulator is not None:
            self.simulator.close()

    # def __del__(self):
    #     """Close the environment when object is deleted."""
    #     self.close()

    def render(self):
        """Render the environment.
        
        If render_mode is "human", the environment will be rendered in a GUI window.
        If render_mode is "rgb_array", returns an RGB array of the current frame.
        """
        # Rendering is handled by the simulator
        if self.render_mode == "human":
            return  # GUI already rendering
        elif self.render_mode == "rgb_array":
            return self.simulator.get_rgb_array()

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file.

        Args:
            out_csv_name (str): Path to the output .csv file. E.g.: "results/my_results
            episode (int): Episode number to be appended to the output file name.
        """
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + f"_conn{self.label}_ep{episode}" + ".csv", index=False)

    def enable_debug_logging(self, enable: bool = True, level: int = 1, ts_ids: list = None):
        """Enable or disable debug logging for traffic signals.
        
        This is useful for debugging reward and action calculations during training.
        
        Args:
            enable: True to enable logging, False to disable
            level: Logging detail level:
                   1 = Basic (reward values, action ratios)
                   2 = Detailed (+ traffic state info)
                   3 = Verbose (+ detector history)
            ts_ids: Optional list of specific traffic signal IDs.
                   If None, enables for all traffic signals.
        
        Example:
            env.enable_debug_logging(True, level=2)  # Enable detailed logging
            env.enable_debug_logging(True, level=1, ts_ids=['tl_1'])  # Enable for specific signal
        """
        self.simulator.enable_debug_logging(enable, level, ts_ids)

    # Below functions are for discrete state space
    # TODO: Update API to support these functions if needed
    # def encode(self, state, ts_id):
    #     """Encode the state of the traffic signal into a hashable object."""
    #     phase = int(np.where(state[: self.traffic_signals[ts_id].num_green_phases] == 1)[0])
    #     min_green = state[self.traffic_signals[ts_id].num_green_phases]
    #     density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1 :]]
    #     # tuples are hashable and can be used as key in python dictionary
    #     return tuple([phase, min_green] + density_queue)

    # def _discretize_density(self, density):
    #     return min(int(density * 10), 9)


class SumoEnvironmentPZ(AECEnv, EzPickle):
    """A wrapper for the SUMO environment that implements the AECEnv interface from PettingZoo.

    For more information, see https://pettingzoo.farama.org/api/aec/.

    The arguments are the same as for :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "name": "sumo_rl_v0", "is_parallelizable": True}

    def __init__(self, **kwargs):
        """Initialize the environment."""
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = SumoEnvironment(**self._kwargs)
        self.render_mode = self.env.render_mode

        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def seed(self, seed=None):
        """Set the seed for the environment."""
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.compute_info()

    def compute_info(self):
        """Compute the info for the current step."""
        self.infos = {a: {} for a in self.agents}
        infos = self.env._compute_info()
        for a in self.agents:
            for k, v in infos.items():
                if k.startswith(a) or k.startswith("system"):
                    self.infos[a][k] = v

    def observation_space(self, agent):
        """Return the observation space for the agent."""
        return self.env.simulator.get_observation_space(agent)

    def action_space(self, agent):
        """Return the action space for the agent."""
        return self.env.simulator.get_action_space(agent)

    def observe(self, agent):
        """Return the observation for the agent."""
        obs = self.env.observations[agent].copy()
        return obs

    def close(self):
        """Close the environment."""
        self.env.simulator.close()

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            return  # GUI already rendering
        elif self.render_mode == "rgb_array":
            return self.env.simulator.get_rgb_array()

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file."""
        if out_csv_name:
            df = pd.DataFrame(self.env.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + f"_ep{episode}" + ".csv", index=False)

    def step(self, action):
        """Step the environment with the given action."""
        # Check if agent is done
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)

        agent = self.agent_selection
        
        # Validate action
        if not self.action_spaces[agent].contains(action):
            raise Exception(
                "Action for agent {} must be in Discrete({})."
                "It is currently {}".format(agent, self.action_spaces[agent].n, action)
            )

        # Step simulator if this is the last agent
        if self._agent_selector.is_last():
            # Convert single agent action to dict
            actions = {agent: action} if not self.env.fixed_ts else {}
            
            # Step simulator and get results
            observations, rewards, dones, info = self.env.simulator.step(actions)
            
            # Update environment state
            self.observations = observations
            self.rewards = rewards
            self.compute_info()
            
            # Check if episode is done
            done = dones["__all__"]
            self.truncations = {a: done for a in self.agents}
        else:
            self._clear_rewards()

        # Update agent selection
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()