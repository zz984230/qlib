#!/usr/bin/env python
"""View and analyze strategies from the strategy pool.

Usage:
    python view_strategies.py --symbol 601138
    python view_strategies.py --symbol 601138 --top 5
    python view_strategies.py --stats
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="View strategy pool")
    parser.add_argument("--symbol", help="Filter by stock symbol")
    parser.add_argument("--top", type=int, default=10, help="Show top N strategies")
    parser.add_argument("--stats", action="store_true", help="Show pool statistics")
    parser.add_argument("--sort-by", choices=["fitness", "return", "sharpe", "drawdown"],
                        default="fitness", help="Sort strategies by")
    parser.add_argument("--export", help="Export to CSV file")
    return parser.parse_args()


def load_strategies(pool_dir: Path) -> list[dict]:
    """Load all strategies from pool."""
    strategies = []
    for file_path in pool_dir.glob("strategy_*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                strategies.append(json.load(f))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return strategies


def get_sort_key(strategy: dict, sort_by: str) -> float:
    """Get sort key for strategy."""
    if sort_by == "fitness":
        return strategy.get("fitness", 0)
    elif sort_by == "return":
        return strategy.get("backtest_results", {}).get("1y", {}).get("total_return", 0)
    elif sort_by == "sharpe":
        return strategy.get("backtest_results", {}).get("1y", {}).get("sharpe_ratio", 0)
    elif sort_by == "drawdown":
        return -strategy.get("backtest_results", {}).get("1y", {}).get("max_drawdown", 1)
    return 0


def format_strategy(strategy: dict, rank: int) -> str:
    """Format strategy for display."""
    backtest = strategy.get("backtest_results", {})
    y1 = backtest.get("1y", {})
    m3 = backtest.get("3m", {})
    m1 = backtest.get("1m", {})

    # Get top 3 factors
    factors = sorted(
        strategy.get("factor_weights", {}).items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    factor_str = ", ".join([f"{k}:{v:.2f}" for k, v in factors])

    return f"""
[{rank}] {strategy.get('id', 'unknown')}
    Symbol: {strategy.get('symbol')}
    Fitness: {strategy.get('fitness', 0):.4f}
    Signal Threshold: {strategy.get('signal_threshold', 0):.2f}
    Exit Threshold: {strategy.get('exit_threshold', 0):.2f}
    Top Factors: {factor_str}

    1Y Return: {y1.get('total_return', 0):.2%} | Drawdown: {y1.get('max_drawdown', 0):.2%} | Sharpe: {y1.get('sharpe_ratio', 0):.2f}
    3M Return: {m3.get('total_return', 0):.2%} | Drawdown: {m3.get('max_drawdown', 0):.2%}
    1M Return: {m1.get('total_return', 0):.2%} | Drawdown: {m1.get('max_drawdown', 0):.2%}
"""


def main():
    args = parse_args()

    # Determine pool directory
    pool_dir = Path("strategies/pool")
    if not pool_dir.exists():
        pool_dir = Path(__file__).parent.parent.parent.parent / "strategies" / "pool"

    if not pool_dir.exists():
        print("Strategy pool directory not found")
        return 1

    # Load strategies
    strategies = load_strategies(pool_dir)

    if not strategies:
        print("No strategies found in pool")
        return 0

    # Filter by symbol
    if args.symbol:
        strategies = [s for s in strategies if s.get("symbol") == args.symbol]

    if args.stats:
        # Show statistics
        symbols = {}
        for s in strategies:
            sym = s.get("symbol", "unknown")
            symbols[sym] = symbols.get(sym, 0) + 1

        fitness_values = [s.get("fitness", 0) for s in strategies]

        print("\n=== Strategy Pool Statistics ===")
        print(f"Total Strategies: {len(strategies)}")
        print(f"Symbols: {symbols}")
        print(f"Best Fitness: {max(fitness_values):.4f}")
        print(f"Average Fitness: {sum(fitness_values) / len(fitness_values):.4f}")
        return 0

    # Sort strategies
    strategies.sort(key=lambda s: get_sort_key(s, args.sort_by), reverse=True)

    # Show top N
    print(f"\n=== Top {args.top} Strategies (sorted by {args.sort_by}) ===")

    for i, strategy in enumerate(strategies[:args.top], 1):
        print(format_strategy(strategy, i))

    # Export to CSV if requested
    if args.export:
        import csv
        with open(args.export, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "symbol", "fitness", "signal_threshold", "exit_threshold",
                "1y_return", "1y_drawdown", "1y_sharpe",
                "3m_return", "3m_drawdown",
                "1m_return", "1m_drawdown"
            ])
            for s in strategies:
                y1 = s.get("backtest_results", {}).get("1y", {})
                m3 = s.get("backtest_results", {}).get("3m", {})
                m1 = s.get("backtest_results", {}).get("1m", {})
                writer.writerow([
                    s.get("id"),
                    s.get("symbol"),
                    s.get("fitness"),
                    s.get("signal_threshold"),
                    s.get("exit_threshold"),
                    y1.get("total_return"),
                    y1.get("max_drawdown"),
                    y1.get("sharpe_ratio"),
                    m3.get("total_return"),
                    m3.get("max_drawdown"),
                    m1.get("total_return"),
                    m1.get("max_drawdown"),
                ])
        print(f"Exported to {args.export}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
