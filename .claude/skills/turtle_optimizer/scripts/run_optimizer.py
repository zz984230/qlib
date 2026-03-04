#!/usr/bin/env python
"""Run turtle genetic optimizer with common configurations.

Usage:
    python run_optimizer.py --symbol 601138 --name "Stock Name"
    python run_optimizer.py --symbol 601138 --name "Stock Name" --quick
    python run_optimizer.py --symbol 601138 --name "Stock Name" --thorough
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run turtle genetic optimizer")
    parser.add_argument("--symbol", required=True, help="Stock symbol (e.g., 601138)")
    parser.add_argument("--name", required=True, help="Stock name")
    parser.add_argument("--cash", type=float, default=50000, help="Initial capital")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer generations)")
    parser.add_argument("--thorough", action="store_true", help="Thorough run (more generations)")
    parser.add_argument("--no-interactive", action="store_true", help="Non-interactive mode")
    parser.add_argument("--max-drawdown", type=float, default=0.03, help="Max drawdown limit")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine configuration based on mode
    if args.quick:
        generations = 30
        population = 30
    elif args.thorough:
        generations = 200
        population = 100
    else:
        generations = 100
        population = 50

    # Build command
    cmd = [
        sys.executable,
        "scripts/turtle_genetic_optimizer.py",
        "--symbol", args.symbol,
        "--name", args.name,
        "--cash", str(args.cash),
        "--max-generations", str(generations),
        "--population-size", str(population),
        "--max-drawdown", str(args.max_drawdown),
    ]

    if args.no_interactive:
        cmd.append("--no-interactive")

    # Run optimizer
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent.parent)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
