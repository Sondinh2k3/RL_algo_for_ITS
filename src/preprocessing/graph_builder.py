"""
Graph Builder Module for Directional Adjacency Matrix.

This module creates directional adjacency matrices for graph neural networks,
enabling proper directional information flow in traffic networks.

The directional adjacency matrix is a [4, N, N] tensor where:
- Channel 0: North neighbors (neighbors located to the north of each node)
- Channel 1: East neighbors (neighbors located to the east of each node)
- Channel 2: South neighbors (neighbors located to the south of each node)
- Channel 3: West neighbors (neighbors located to the west of each node)
"""

import numpy as np
import math
import torch
from typing import Dict, List, Tuple, Optional, Union
import xml.etree.ElementTree as ET


def get_direction_index(angle_degrees: float) -> int:
    """
    Convert angle (in degrees) to direction index.
    
    SUMO coordinates: Y increases upward (North), X increases rightward (East).
    
    Args:
        angle_degrees: Angle in degrees from atan2(dy, dx)
        
    Returns:
        Direction index: 0=North, 1=East, 2=South, 3=West
    """
    # Normalize angle to [0, 360)
    angle = angle_degrees % 360
    
    # Direction classification:
    # North: 45 to 135 degrees
    # East: 315 to 360 or 0 to 45 degrees  
    # South: 225 to 315 degrees
    # West: 135 to 225 degrees
    
    if 45 <= angle < 135:
        return 0  # North
    elif 135 <= angle < 225:
        return 3  # West
    elif 225 <= angle < 315:
        return 2  # South
    else:  # 315-360 or 0-45
        return 1  # East


def build_directional_adjacency(
    edge_list: List[Tuple[int, int]],
    node_coords: Dict[int, Tuple[float, float]],
    num_nodes: int,
    bidirectional: bool = True
) -> np.ndarray:
    """
    Build directional adjacency matrices from edge list and node coordinates.
    
    Args:
        edge_list: List of (source_node_idx, target_node_idx) tuples
        node_coords: Dictionary mapping node index to (x, y) coordinates
        num_nodes: Total number of nodes
        bidirectional: If True, add reverse edges automatically
        
    Returns:
        adj_stack: np.ndarray of shape [4, num_nodes, num_nodes]
                   Channel 0: North neighbors
                   Channel 1: East neighbors
                   Channel 2: South neighbors
                   Channel 3: West neighbors
                   
    Example:
        If node B is located to the North of node A, then:
        adj_stack[0, A, B] = 1.0  (B is a North neighbor of A)
    """
    adj_stack = np.zeros((4, num_nodes, num_nodes), dtype=np.float32)
    
    edges_to_process = list(edge_list)
    if bidirectional:
        # Add reverse edges
        for u, v in edge_list:
            if (v, u) not in edges_to_process:
                edges_to_process.append((v, u))
    
    for u, v in edges_to_process:
        if u >= num_nodes or v >= num_nodes:
            continue
        if u not in node_coords or v not in node_coords:
            continue
            
        pos_u = node_coords[u]
        pos_v = node_coords[v]  # v is neighbor of u
        
        # Vector from u to v
        dx = pos_v[0] - pos_u[0]
        dy = pos_v[1] - pos_u[1]
        
        # Skip if same position
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        
        # Calculate angle in degrees
        angle = math.degrees(math.atan2(dy, dx))
        
        # Get direction index
        direction_idx = get_direction_index(angle)
        
        # v is a [direction] neighbor of u
        adj_stack[direction_idx, u, v] = 1.0

    return adj_stack


def build_directional_adjacency_from_net_file(
    net_file: str,
    ts_ids: List[str],
    ts_id_to_idx: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Build directional adjacency matrix directly from SUMO .net.xml file.
    
    Args:
        net_file: Path to SUMO network file (.net.xml)
        ts_ids: List of traffic signal IDs (junction IDs) to include
        ts_id_to_idx: Optional mapping from ts_id to node index. 
                      If None, uses order in ts_ids.
                      
    Returns:
        adj_stack: np.ndarray of shape [4, num_nodes, num_nodes]
    """
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    num_nodes = len(ts_ids)
    
    # Create mapping if not provided
    if ts_id_to_idx is None:
        ts_id_to_idx = {ts_id: idx for idx, ts_id in enumerate(ts_ids)}
    
    # Get junction coordinates
    node_coords = {}
    for junction in root.findall('junction'):
        junc_id = junction.get('id')
        if junc_id in ts_id_to_idx:
            x = float(junction.get('x', 0))
            y = float(junction.get('y', 0))
            node_coords[ts_id_to_idx[junc_id]] = (x, y)
    
    # Build edge list from edges connecting traffic signals
    edge_list = []
    ts_id_set = set(ts_ids)
    
    for edge in root.findall('edge'):
        # Skip internal edges
        if edge.get('function') == 'internal':
            continue
            
        from_junc = edge.get('from')
        to_junc = edge.get('to')
        
        # Only include edges between traffic signals
        if from_junc in ts_id_set and to_junc in ts_id_set:
            u = ts_id_to_idx[from_junc]
            v = ts_id_to_idx[to_junc]
            if (u, v) not in edge_list:
                edge_list.append((u, v))
    
    # Build directional adjacency
    return build_directional_adjacency(
        edge_list=edge_list,
        node_coords=node_coords,
        num_nodes=num_nodes,
        bidirectional=True
    )


def adjacency_to_tensor(
    adj_stack: np.ndarray,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert numpy adjacency stack to PyTorch tensor.
    
    Args:
        adj_stack: np.ndarray of shape [4, N, N]
        device: Target device for tensor
        
    Returns:
        torch.Tensor of shape [4, N, N]
    """
    tensor = torch.from_numpy(adj_stack)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def expand_adjacency_for_batch(
    adj_directions: torch.Tensor,
    batch_size: int
) -> torch.Tensor:
    """
    Expand directional adjacency for batch processing.
    
    Args:
        adj_directions: [4, N, N] tensor
        batch_size: Batch size
        
    Returns:
        [Batch, 4, N, N] tensor
    """
    return adj_directions.unsqueeze(0).expand(batch_size, -1, -1, -1)


# Legacy function for backward compatibility
def build_simple_adjacency(
    edge_list: List[Tuple[int, int]],
    num_nodes: int,
    bidirectional: bool = True
) -> np.ndarray:
    """
    Build simple (non-directional) adjacency matrix.
    
    Args:
        edge_list: List of (source, target) tuples
        num_nodes: Number of nodes
        bidirectional: If True, add reverse edges
        
    Returns:
        adj: np.ndarray of shape [N, N]
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for u, v in edge_list:
        if u < num_nodes and v < num_nodes:
            adj[u, v] = 1.0
            if bidirectional:
                adj[v, u] = 1.0
                
    return adj
