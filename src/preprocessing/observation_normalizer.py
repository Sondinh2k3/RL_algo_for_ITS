"""Running statistics for observation and reward normalization.

This module provides utilities for online normalization of observations and rewards
using running mean and standard deviation estimates.
"""

import numpy as np
from typing import Tuple, Optional


class RunningMeanStd:
    """Tracks the running mean and standard deviation of a data stream.
    
    Uses Welford's online algorithm for numerical stability.
    Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    
    This is commonly used in RL for:
    - Observation normalization: Normalize observations to have mean=0, std=1
    - Reward normalization: Normalize rewards for more stable value function learning
    
    Example:
        >>> rms = RunningMeanStd(shape=(4,))
        >>> for obs in observations:
        ...     rms.update(obs)
        ...     normalized_obs = (obs - rms.mean) / np.sqrt(rms.var + 1e-8)
    """
    
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        """Initialize running statistics.
        
        Args:
            shape: Shape of the data (e.g., () for scalar, (4,) for 4D vector)
            epsilon: Small constant for numerical stability in variance
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Small value to avoid division by zero
        self._epsilon = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new data.
        
        Uses Welford's online algorithm for numerical stability.
        
        Args:
            x: New data point(s). Can be a single sample or a batch.
               For batch, assumes first dimension is batch dimension.
        """
        x = np.asarray(x, dtype=np.float64)
        
        # Skip update if input contains NaN or Inf
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return
        
        # Handle batch updates
        if x.shape == self.mean.shape:
            # Single sample
            batch_mean = x
            batch_var = np.zeros_like(self.var)
            batch_count = 1
        else:
            # Batch of samples - compute batch statistics
            # Skip if batch is too small or has no variance
            if len(x) < 2:
                batch_mean = np.mean(x, axis=0) if len(x) > 0 else self.mean
                batch_var = np.zeros_like(self.var)
                batch_count = len(x)
            else:
                batch_mean = np.mean(x, axis=0)
                batch_var = np.var(x, axis=0)
                batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self, 
        batch_mean: np.ndarray, 
        batch_var: np.ndarray, 
        batch_count: int
    ) -> None:
        """Update statistics from batch moments using parallel algorithm.
        
        Reference: Chan et al. (1979) - Updating Formulae and a Pairwise Algorithm
        for Computing Sample Variances
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # Update mean
        new_mean = self.mean + delta * batch_count / total_count
        
        # Update variance using parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x: np.ndarray, clip: Optional[float] = None) -> np.ndarray:
        """Normalize data using current running statistics.
        
        Args:
            x: Data to normalize
            clip: If provided, clip normalized values to [-clip, clip]
            
        Returns:
            Normalized data with approximately mean=0, std=1
        """
        x = np.asarray(x, dtype=np.float32)
        normalized = (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)
        
        if clip is not None:
            normalized = np.clip(normalized, -clip, clip)
        
        return normalized
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale.
        
        Args:
            x: Normalized data
            
        Returns:
            Data in original scale
        """
        x = np.asarray(x, dtype=np.float32)
        return x * np.sqrt(self.var.astype(np.float32) + 1e-8) + self.mean.astype(np.float32)
    
    def reset(self) -> None:
        """Reset statistics to initial state."""
        self.mean = np.zeros_like(self.mean)
        self.var = np.ones_like(self.var)
        self.count = self._epsilon
    
    def get_state(self) -> dict:
        """Get current state for serialization.
        
        Returns:
            Dictionary containing mean, var, and count
        """
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count
        }
    
    def set_state(self, state: dict) -> None:
        """Restore state from serialized data.
        
        Args:
            state: Dictionary from get_state()
        """
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]


class RewardNormalizer:
    """Specialized normalizer for rewards with additional features.
    
    Features:
    - Running return normalization (optional)
    - Per-agent normalization support
    - Reward clipping
    
    This follows the approach used in OpenAI baselines and Stable-Baselines3.
    """
    
    def __init__(
        self, 
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: Optional[float] = 10.0,
        per_agent: bool = False,
        num_agents: int = 1
    ):
        """Initialize reward normalizer.
        
        Args:
            gamma: Discount factor for return normalization
            epsilon: Small constant for numerical stability
            clip: Clip normalized rewards to [-clip, clip]. None to disable.
            per_agent: If True, maintain separate statistics per agent
            num_agents: Number of agents (only used if per_agent=True)
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.per_agent = per_agent
        
        if per_agent:
            self.rms = {i: RunningMeanStd(shape=()) for i in range(num_agents)}
            self.returns = {i: 0.0 for i in range(num_agents)}
        else:
            self.rms = RunningMeanStd(shape=())
            self.returns = 0.0
    
    def normalize(self, rewards: dict, dones: Optional[dict] = None) -> dict:
        """Normalize rewards using running statistics.
        
        Args:
            rewards: Dictionary mapping agent_id -> reward
            dones: Optional dictionary mapping agent_id -> done flag
            
        Returns:
            Dictionary of normalized rewards
        """
        normalized = {}
        
        if self.per_agent:
            for agent_id, reward in rewards.items():
                idx = list(rewards.keys()).index(agent_id)
                if idx in self.rms:
                    rms = self.rms[idx]
                else:
                    rms = self.rms[0]  # Fallback
                
                # Update statistics
                rms.update(np.array([reward]))
                
                # Normalize
                norm_reward = reward / np.sqrt(rms.var + self.epsilon)
                
                if self.clip is not None:
                    norm_reward = np.clip(norm_reward, -self.clip, self.clip)
                
                normalized[agent_id] = float(norm_reward)
        else:
            # Collect all rewards for batch update
            reward_values = np.array(list(rewards.values()), dtype=np.float64)
            
            # Update running statistics with batch
            self.rms.update(reward_values)
            
            # Normalize each reward using shared statistics
            for agent_id, reward in rewards.items():
                norm_reward = (reward - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
                
                if self.clip is not None:
                    norm_reward = np.clip(norm_reward, -self.clip, self.clip)
                
                normalized[agent_id] = float(norm_reward)
        
        return normalized
    
    def reset(self) -> None:
        """Reset normalizer state."""
        if self.per_agent:
            for rms in self.rms.values():
                rms.reset()
            self.returns = {i: 0.0 for i in self.returns.keys()}
        else:
            self.rms.reset()
            self.returns = 0.0
