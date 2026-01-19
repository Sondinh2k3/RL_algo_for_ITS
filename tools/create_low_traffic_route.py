import xml.etree.ElementTree as ET
import sys
import os
from pathlib import Path

def create_low_traffic_route(input_file, output_file, keep_ratio=0.1):
    """Creates a low traffic route file by sampling trips."""
    print(f"Reading {input_file}...")
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Create new root
    new_root = ET.Element("routes")
    new_root.attrib = root.attrib
    
    # Copy vTypes
    for child in root:
        if child.tag in ["vType", "vTypeDistribution"]:
            new_root.append(child)
            
    # Copy trips with sampling
    trips = [child for child in root if child.tag == "trip"]
    print(f"Found {len(trips)} trips.")
    
    kept_trips = 0
    for i, trip in enumerate(trips):
        if i % int(1/keep_ratio) == 0:
            new_root.append(trip)
            kept_trips += 1
            
    print(f"Kept {kept_trips} trips ({kept_trips/len(trips)*100:.1f}%).")
    
    # Write to file
    tree = ET.ElementTree(new_root)
    ET.indent(tree, space="    ", level=0)
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)
    print(f"Wrote to {output_file}")

if __name__ == "__main__":
    base_path = Path("network/grid4x4")
    input_file = base_path / "grid4x4-demo.rou.xml"
    output_file = base_path / "grid4x4-low-traffic.rou.xml"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        sys.exit(1)
        
    create_low_traffic_route(input_file, output_file, keep_ratio=0.1)
