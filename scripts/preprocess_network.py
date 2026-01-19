#!/usr/bin/env python3
"""Network Preprocessing Script - Generate standardized intersection configuration.

This script runs GPI and FRAP preprocessing on a SUMO network to create
a JSON configuration file that can be used by the MGMQ training algorithm.

Usage:
    # Preprocess all traffic lights in network
    python scripts/preprocess_network.py --network grid4x4
    
    # Preprocess only specific intersections
    python scripts/preprocess_network.py --network grid4x4 --ts-ids A0 A1 B0 B1
    
    python scripts/preprocess_network.py --config src/config/simulation.yml
    
Output:
    Creates intersection_config.json in the network directory with:
    - Direction mappings (N/E/S/W) for each intersection
    - Lane aggregation info (merged lanes, missing lane masks)
    - Phase standardization (movement-based phases)
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import yaml
import sumolib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import IntersectionStandardizer, PhaseStandardizer


def parse_detector_file(detector_file: str) -> Dict[str, Any]:
    """Parse detector.add.xml to get lane-to-detector mapping.
    
    Args:
        detector_file: Path to detector.add.xml file
        
    Returns:
        Dict with keys:
        - 'lane_to_e1': Dict[lane_id, List[detector_id]]
        - 'lane_to_e2': Dict[lane_id, List[detector_id]]
        - 'e1_detectors': List of all E1 detector IDs
        - 'e2_detectors': List of all E2 detector IDs
        - 'e1_lengths': Dict[detector_id, length]
        - 'e2_lengths': Dict[detector_id, length]
    """
    result = {
        'lane_to_e1': defaultdict(list),
        'lane_to_e2': defaultdict(list),
        'e1_detectors': [],
        'e2_detectors': [],
        'e1_lengths': {},
        'e2_lengths': {},
    }
    
    if not os.path.exists(detector_file):
        print(f"Warning: Detector file not found: {detector_file}")
        return result
    
    try:
        tree = ET.parse(detector_file)
        root = tree.getroot()
        
        # Parse E1 detectors (inductionLoop)
        for det in root.findall('.//inductionLoop'):
            det_id = det.get('id')
            lane_id = det.get('lane')
            if det_id and lane_id:
                result['lane_to_e1'][lane_id].append(det_id)
                result['e1_detectors'].append(det_id)
        
        # Parse E2 detectors (laneAreaDetector)
        for det in root.findall('.//laneAreaDetector'):
            det_id = det.get('id')
            lane_id = det.get('lane')
            length = det.get('length', '0')
            if det_id and lane_id:
                result['lane_to_e2'][lane_id].append(det_id)
                result['e2_detectors'].append(det_id)
                try:
                    result['e2_lengths'][det_id] = float(length)
                except ValueError:
                    result['e2_lengths'][det_id] = 60.0  # Default length
        
        # Convert defaultdicts to regular dicts
        result['lane_to_e1'] = dict(result['lane_to_e1'])
        result['lane_to_e2'] = dict(result['lane_to_e2'])
        
        print(f"   Parsed {len(result['e1_detectors'])} E1 detectors, {len(result['e2_detectors'])} E2 detectors")
        
    except ET.ParseError as e:
        print(f"Error parsing detector file: {e}")
    
    return result


class NetworkDataProvider:
    """Data provider for preprocessing that reads directly from SUMO network files.
    
    This is used during preprocessing (before simulation starts) to extract
    network topology and lane information from .net.xml files.
    """
    
    def __init__(self, net_file: str):
        """Initialize with SUMO network file.
        
        Args:
            net_file: Path to .net.xml file
        """
        self.net = sumolib.net.readNet(net_file, withPrograms=True)
        self._lane_shapes = {}
        self._incoming_edges = {}
        
    def get_incoming_edges(self, junction_id: str) -> List[str]:
        """Get incoming edge IDs for a junction."""
        if junction_id in self._incoming_edges:
            return self._incoming_edges[junction_id]
            
        junction = self.net.getNode(junction_id)
        if junction is None:
            return []
            
        edges = [e.getID() for e in junction.getIncoming()]
        self._incoming_edges[junction_id] = edges
        return edges
    
    def get_outgoing_edges(self, junction_id: str) -> List[str]:
        """Get outgoing edge IDs for a junction."""
        junction = self.net.getNode(junction_id)
        if junction is None:
            return []
        return [e.getID() for e in junction.getOutgoing()]
    
    def get_lane_shape(self, lane_id: str) -> List[tuple]:
        """Get lane shape (coordinates) from network."""
        if lane_id in self._lane_shapes:
            return self._lane_shapes[lane_id]
            
        try:
            edge_id = lane_id.rsplit('_', 1)[0]
            lane_index = int(lane_id.rsplit('_', 1)[1])
            edge = self.net.getEdge(edge_id)
            lane = edge.getLane(lane_index)
            shape = lane.getShape()
            self._lane_shapes[lane_id] = shape
            return shape
        except Exception:
            return []
    
    def get_edge_lanes(self, edge_id: str) -> List[str]:
        """Get all lane IDs for an edge."""
        try:
            edge = self.net.getEdge(edge_id)
            return [lane.getID() for lane in edge.getLanes()]
        except Exception:
            return []
    
    def get_junction_tls(self, junction_id: str) -> Optional[str]:
        """Get traffic light ID for a junction (if exists)."""
        junction = self.net.getNode(junction_id)
        if junction is None:
            return None
        tls = junction.getTLS()
        if tls:
            return tls.getID()
        return None
    
    def get_all_tls_ids(self) -> List[str]:
        """Get all traffic light IDs in the network."""
        return [tls.getID() for tls in self.net.getTrafficLights()]

    def get_traffic_light_program(self, tls_id: str) -> Any:
        """Get traffic light program logic."""
        tls = self.net.getTLSSecure(tls_id)
        if not tls:
            return None
        programs = tls.getPrograms()
        if not programs:
            return None
        # Return the first program (usually '0')
        return list(programs.values())[0]

    def get_controlled_links(self, tls_id: str) -> List[List[Any]]:
        """Get controlled links for the traffic light.
        
        Returns:
            List of lists of connections, grouped by link index.
        """
        tls = self.net.getTLSSecure(tls_id)
        if not tls:
            return []
        
        links_dict = tls.getLinks() # Dict[int, List[Connection]]
        if not links_dict:
            return []
            
        max_index = max(links_dict.keys())
        result = []
        for i in range(max_index + 1):
            result.append(links_dict.get(i, []))
            
        return result


class LaneAggregator:
    """Handles lane aggregation for standardized observation space.
    
    This module ensures all intersections have the same observation dimension
    by merging extra lanes and masking missing lanes.
    """
    
    def __init__(
        self,
        standard_lanes_per_direction: int = 2,
        missing_lane_strategy: str = "zero",
        merge_strategy: str = "mean"
    ):
        """Initialize lane aggregator.
        
        Args:
            standard_lanes_per_direction: Target number of lanes per direction
            missing_lane_strategy: How to handle missing lanes ("zero" or "mask")
            merge_strategy: How to merge extra lanes ("sum", "mean", "max")
        """
        self.standard_lanes = standard_lanes_per_direction
        self.missing_strategy = missing_lane_strategy
        self.merge_strategy = merge_strategy
    
    def compute_aggregation(
        self,
        lanes_by_direction: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Compute lane aggregation configuration.
        
        Args:
            lanes_by_direction: Dict mapping direction (N/E/S/W) to lane IDs
            
        Returns:
            Dict with aggregation configuration for each direction
        """
        result = {}
        
        for direction in ['N', 'E', 'S', 'W']:
            lanes = lanes_by_direction.get(direction, [])
            num_lanes = len(lanes)
            
            direction_config = {
                "original_lanes": lanes,
                "num_original_lanes": num_lanes,
                "num_standard_lanes": self.standard_lanes,
            }
            
            if num_lanes == 0:
                # Missing direction entirely
                direction_config["status"] = "missing"
                direction_config["mask"] = [0] * self.standard_lanes
                direction_config["aggregation_indices"] = []
                
            elif num_lanes == self.standard_lanes:
                # Perfect match
                direction_config["status"] = "exact"
                direction_config["mask"] = [1] * self.standard_lanes
                direction_config["aggregation_indices"] = [[i] for i in range(num_lanes)]
                
            elif num_lanes < self.standard_lanes:
                # Fewer lanes than standard - pad with zeros/mask
                direction_config["status"] = "padded"
                direction_config["mask"] = [1] * num_lanes + [0] * (self.standard_lanes - num_lanes)
                direction_config["aggregation_indices"] = [[i] for i in range(num_lanes)]
                # Pad with empty indices
                direction_config["aggregation_indices"] += [[] for _ in range(self.standard_lanes - num_lanes)]
                
            else:
                # More lanes than standard - need to merge
                direction_config["status"] = "merged"
                direction_config["mask"] = [1] * self.standard_lanes
                direction_config["merge_strategy"] = self.merge_strategy
                
                # Distribute lanes across standard slots
                lanes_per_slot = num_lanes / self.standard_lanes
                indices = []
                for i in range(self.standard_lanes):
                    start = int(i * lanes_per_slot)
                    end = int((i + 1) * lanes_per_slot)
                    indices.append(list(range(start, end)))
                direction_config["aggregation_indices"] = indices
            
            result[direction] = direction_config
        
        return result


def preprocess_network(
    net_file: str,
    output_file: str,
    config: Dict[str, Any],
    selected_ts_ids: Optional[List[str]] = None,
    detector_file: Optional[str] = None
) -> Dict[str, Any]:
    """Preprocess a SUMO network and generate configuration.
    
    Args:
        net_file: Path to .net.xml file
        output_file: Path to output JSON file
        config: Preprocessing configuration
        selected_ts_ids: Optional list of specific traffic signal IDs to process.
                        If None, all traffic lights in the network will be processed.
        
    Returns:
        Generated configuration dict
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING NETWORK FOR MGMQ")
    print(f"{'='*60}")
    print(f"Network file: {net_file}")
    print(f"Output file: {output_file}")
    
    # Initialize data provider
    data_provider = NetworkDataProvider(net_file)
    
    # Get traffic light IDs to process
    all_tls_ids = data_provider.get_all_tls_ids()
    
    if selected_ts_ids:
        # Validate selected IDs exist in network
        invalid_ids = [ts_id for ts_id in selected_ts_ids if ts_id not in all_tls_ids]
        if invalid_ids:
            print(f"\n⚠ Warning: The following IDs not found in network: {invalid_ids}")
            print(f"  Available IDs: {all_tls_ids}")
        
        # Filter to only valid selected IDs
        tls_ids = [ts_id for ts_id in selected_ts_ids if ts_id in all_tls_ids]
        print(f"\nSelected {len(tls_ids)} traffic lights: {tls_ids}")
        print(f"(Network has {len(all_tls_ids)} total traffic lights)")
    else:
        tls_ids = all_tls_ids
        print(f"\nProcessing all {len(tls_ids)} traffic lights: {tls_ids}")
    
    # Initialize lane aggregator
    gpi_config = config.get("gpi", {})
    lane_config = gpi_config.get("lane_aggregation", {})
    
    aggregator = LaneAggregator(
        standard_lanes_per_direction=lane_config.get("standard_lanes_per_direction", 2),
        missing_lane_strategy=lane_config.get("missing_lane_strategy", "zero"),
        merge_strategy=lane_config.get("merge_strategy", "mean")
    )
    
    # Parse detector file if provided
    detector_info = {'lane_to_e1': {}, 'lane_to_e2': {}, 'e2_lengths': {}}
    if detector_file:
        print(f"\nParsing detector file: {detector_file}")
        detector_info = parse_detector_file(detector_file)
    
    # Process each intersection
    result = {
        "network_file": str(net_file),
        "detector_file": detector_file,
        "num_intersections": len(tls_ids),
        "intersections": {},
        "adjacency": {},
        "config": config,
    }
    
    print(f"\n{'─'*40}")
    print("Processing intersections...")
    print(f"{'─'*40}")
    
    cnt = 0

    for tls_id in tls_ids:
        print(f"\n{cnt}. {tls_id}")
        cnt += 1
        
        # Run GPI - Direction standardization
        gpi = IntersectionStandardizer(tls_id, data_provider=data_provider)
        direction_map = gpi.map_intersection()
        
        print(f"   Directions: N={direction_map['N']}, E={direction_map['E']}, "
              f"S={direction_map['S']}, W={direction_map['W']}")
        
        # Get lanes by direction
        lanes_by_direction = {}
        for direction, edge_id in direction_map.items():
            if edge_id is not None:
                lanes_by_direction[direction] = data_provider.get_edge_lanes(edge_id)
            else:
                lanes_by_direction[direction] = []
        
        # Compute lane aggregation
        aggregation = aggregator.compute_aggregation(lanes_by_direction)
        
        for direction, agg_info in aggregation.items():
            status = agg_info["status"]
            num_orig = agg_info["num_original_lanes"]
            print(f"   {direction}: {num_orig} lanes → {status}")
        
        # Map detectors to directions
        detectors_by_direction = {}
        all_e1_detectors = []
        all_e2_detectors = []
        e2_detector_lengths = {}
        
        for direction in ['N', 'E', 'S', 'W']:
            lanes = lanes_by_direction.get(direction, [])
            direction_e1 = []
            direction_e2 = []
            
            for lane_id in lanes:
                # Get E1 detectors for this lane
                e1_dets = detector_info['lane_to_e1'].get(lane_id, [])
                direction_e1.extend(e1_dets)
                all_e1_detectors.extend(e1_dets)
                
                # Get E2 detectors for this lane
                e2_dets = detector_info['lane_to_e2'].get(lane_id, [])
                direction_e2.extend(e2_dets)
                all_e2_detectors.extend(e2_dets)
                
                # Store E2 lengths
                for det_id in e2_dets:
                    if det_id in detector_info.get('e2_lengths', {}):
                        e2_detector_lengths[det_id] = detector_info['e2_lengths'][det_id]
            
            detectors_by_direction[direction] = {
                "e1": direction_e1,
                "e2": direction_e2
            }
        
        print(f"   Detectors: {len(all_e1_detectors)} E1, {len(all_e2_detectors)} E2")
            
        # Run FRAP - Phase standardization
        frap = PhaseStandardizer(tls_id, gpi_standardizer=gpi, data_provider=data_provider)
        frap.configure()
        
        print(f"   Phases: {frap.num_phases} phases found")
        
        # Store intersection config
        result["intersections"][tls_id] = {
            "direction_map": direction_map,
            "lanes_by_direction": lanes_by_direction,
            "detectors_by_direction": detectors_by_direction,
            "detectors_e1": all_e1_detectors,
            "detectors_e2": all_e2_detectors,
            "e2_detector_lengths": e2_detector_lengths,
            "lane_aggregation": aggregation,
            "observation_mask": gpi.get_observation_mask().tolist(),
            "phase_config": {
                "num_phases": frap.num_phases,
                "phases": [
                    {
                        "index": p.phase_id,
                        "state": p.state,
                        "duration": p.duration,
                        "green_indices": p.green_indices
                    } for p in frap.phases
                ],
                "movement_to_phase": {
                    str(m): p for m, p in frap.movement_to_phase.items()
                },
                "phase_to_movements": {
                    str(p): [str(m) for m in ms] 
                    for p, ms in frap.phase_to_movements.items()
                },
                "actual_to_standard": frap.actual_to_standard,
                "standard_to_actual": frap.standard_to_actual
            }
        }
    
    # Build adjacency matrix
    print(f"\n{'─'*40}")
    print("Building adjacency matrix...")
    print(f"{'─'*40}")
    
    adjacency = {}
    for tls_id in tls_ids:
        neighbors = []
        intersection_info = result["intersections"][tls_id]
        
        for direction, edge_id in intersection_info["direction_map"].items():
            if edge_id is not None:
                # Get the source junction of this incoming edge
                try:
                    edge = data_provider.net.getEdge(edge_id)
                    source_node = edge.getFromNode()
                    source_tls = source_node.getID() if source_node else None
                    
                    if source_tls and source_tls in tls_ids and source_tls != tls_id:
                        neighbors.append({
                            "neighbor_id": source_tls,
                            "direction": direction,
                            "connecting_edge": edge_id
                        })
                except Exception:
                    pass
        
        adjacency[tls_id] = neighbors
        print(f"   {tls_id}: {len(neighbors)} neighbors")
    
    result["adjacency"] = adjacency
    
    # Save to file
    print(f"\n{'─'*40}")
    print(f"Saving to {output_file}...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Configuration saved!")
    print(f"{'='*60}\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SUMO network for MGMQ training"
    )
    parser.add_argument(
        "--network", "-n",
        type=str,
        default="grid4x4",
        help="Network name (e.g., grid4x4, network_test)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to simulation.yml config file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: network/{name}/intersection_config.json)"
    )
    parser.add_argument(
        "--ts-ids", "-t",
        type=str,
        nargs="+",
        default=None,
        help="Specific traffic signal IDs to preprocess (default: all traffic lights)"
    )
    parser.add_argument(
        "--detector-file", "-d",
        type=str,
        default=None,
        help="Path to detector.add.xml file (default: auto-detect in network directory)"
    )
    
    args = parser.parse_args()
    
    # Determine paths
    project_root = Path(__file__).parent.parent
    
    # Load config
    if args.config:
        config_file = Path(args.config)
    else:
        config_file = project_root / "src" / "config" / "simulation.yml"
    
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        preprocessing_config = config.get("preprocessing", {})
    else:
        print(f"Warning: Config file not found: {config_file}")
        preprocessing_config = {
            "gpi": {"enabled": True, "lane_aggregation": {"enabled": True}},
            "frap": {"enabled": True}
        }
    
    # Determine network files
    network_dir = project_root / "network" / args.network
    net_file = network_dir / f"{args.network}.net.xml"
    
    if not net_file.exists():
        print(f"Error: Network file not found: {net_file}")
        sys.exit(1)
    
    # Determine detector file (auto-detect if not specified)
    if args.detector_file:
        detector_file = args.detector_file
    else:
        # Try to find detector.add.xml in network directory
        detector_file = network_dir / "detector.add.xml"
        if not detector_file.exists():
            print(f"Note: No detector file found at {detector_file}")
            detector_file = None
        else:
            detector_file = str(detector_file)
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = network_dir / "intersection_config.json"
    
    # Run preprocessing
    preprocess_network(
        net_file=str(net_file),
        output_file=str(output_file),
        config=preprocessing_config,
        selected_ts_ids=args.ts_ids,
        detector_file=detector_file
    )


if __name__ == "__main__":
    main()
