#!/usr/bin/env python3
import argparse
from fluxion.engine import FluxionEngine

def main():
    parser = argparse.ArgumentParser(description="FLUXION v4 Python CLI")
    parser.add_argument("--config", default="config/default.yaml", help="Config file")
    parser.add_argument("--input", required=True, help="Input circuit (BLIF)")
    parser.add_argument("--output", default="placed.def", help="Output DEF")
    
    args = parser.parse_args()
    
    engine = FluxionEngine(args.config)
    engine.place(args.input, args.output)

if __name__ == "__main__":
    main()
