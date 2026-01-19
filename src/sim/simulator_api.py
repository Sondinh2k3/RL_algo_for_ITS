"""Simulator API - Abstract interface for traffic simulation backends.

This module defines the abstract API that environment uses to interact with simulators.
Any traffic simulation backend (SUMO, CityFlow, etc.) should implement this API.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union


class SimulatorAPI(ABC):
    """Abstract base class for traffic simulators.
    
    This interface defines all methods that a traffic simulator must implement
    to be used with the RL environment. The environment doesn't know or care
    about the specific simulator implementation - it only calls these methods.
    """

    @abstractmethod
    def initialize(self):
        """Initialize simulator and return initial state.
        
        Returns:
            Dict[str, Any]: Initial observations for all agents
        """
        pass

    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """Execute one simulation step with given actions.
        
        Args:
            actions: Dictionary {agent_id: action} for each agent
            
        Returns:
            Tuple of (observations, rewards, dones, info)
            - observations: Dict[agent_id, obs]
            - rewards: Dict[agent_id, reward]
            - dones: Dict[agent_id, done] + {"__all__": overall_done}
            - info: Dict with metadata
        """
        pass

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset simulator to initial state.
        
        Returns:
            Dict[str, Any]: Initial observations for all agents
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up and close simulator."""
        pass

    @abstractmethod
    def get_agent_ids(self) -> List[str]:
        """Get list of agent IDs (traffic signal IDs).
        
        Returns:
            List[str]: List of agent identifiers
        """
        pass

    @abstractmethod
    def get_observation_space(self, agent_id: str):
        """Get observation space for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            gym.spaces.Space: Observation space
        """
        pass

    @abstractmethod
    def get_action_space(self, agent_id: str):
        """Get action space for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            gym.spaces.Space: Action space
        """
        pass

    @abstractmethod
    def get_sim_step(self) -> float:
        """Get current simulation time.
        
        Returns:
            float: Current simulation step/time
        """
        pass
