#!/usr/bin/env python
"""
Step 1: Convert Wind AKR CSV to TFCat JSON

This script reads the raw CSV file and converts it to standardized
TFCat JSON format.

Usage:
    python pipeline/01_csv_to_tfcat.py
"""

from pathlib import Path
import sys

from converter import CSVToTFCatConverter, Config

def main() -> None:
    """Main execution function."""
    
    # Load configuration
    config_path = Path('pipeline/config.yaml')
    config = Config.from_yaml(config_path)
    
    # Get paths from config
    csv_path = Path(config.paths['input_csv'])
    output_path = Path(config.paths['output_json'])
    
    # Check if input exists
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print(f"Please ensure the file exists before running this script.")
        sys.exit(1)
    
    # Create converter
    converter = CSVToTFCatConverter(config)
    
    # Run conversion
    try:
        collection = converter.convert(csv_path, output_path)
        
        print(f"\n{'='*70}")
        print(" Conversion Summary")
        print(f"{'='*70}")
        print(f"Input:  {csv_path}")
        print(f"Output: {output_path}")
        print(f"Bursts: {len(collection.features)}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()