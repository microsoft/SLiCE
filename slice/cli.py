#!/usr/bin/env python3
"""
Command-line interface for SLiCE (Schema Lineage Composite Evaluation).
"""

import argparse
import json
import sys
from typing import Any, Dict

from slice import SchemaLineageEvaluator


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{filepath}': {e}")
        sys.exit(1)


def format_results(results: Dict[str, float]) -> str:
    """Format evaluation results for display."""
    output = []
    output.append("=" * 50)
    output.append("EVALUATION RESULTS")
    output.append("=" * 50)

    # Overall score
    overall = results.get("overall", results.get("all", 0))
    output.append(f"Overall Score: {overall:.4f}")
    output.append(f"Format Correctness: {results.get('format', 0):.4f}")

    # Component scores
    output.append("\nComponent Scores:")
    if "source_schema" in results:
        output.append(f"  Source Schema: {results['source_schema']:.4f}")
    if "source_table" in results:
        output.append(f"  Source Table: {results['source_table']:.4f}")
    if "transformation" in results:
        output.append(f"  Transformation: {results['transformation']:.4f}")
    if "aggregation" in results:
        output.append(f"  Aggregation: {results['aggregation']:.4f}")
    if "metadata" in results:
        output.append(f"  Metadata: {results['metadata']:.4f}")

    return "\n".join(output)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SLiCE: Schema Lineage Composite Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  slice-eval predicted.json ground_truth.json
  slice-eval --metadata predicted.json ground_truth.json
  slice-eval --weights source_table=0.5,transformation=0.3,aggregation=0.2 predicted.json ground_truth.json
        """,
    )

    parser.add_argument(
        "predicted", help="Path to JSON file containing predicted lineage"
    )
    parser.add_argument(
        "ground_truth", help="Path to JSON file containing ground truth lineage"
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Include metadata evaluation (default: False)",
    )
    parser.add_argument(
        "--weights",
        help="Custom weights in format: source_table=0.4,transformation=0.4,aggregation=0.2",
    )
    parser.add_argument(
        "--output", "-o", help="Output file for results (default: stdout)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--version", action="version", version="SLiCE 0.1.0")

    args = parser.parse_args()

    # Load data
    if args.verbose:
        print(f"Loading predicted lineage from: {args.predicted}")
        print(f"Loading ground truth lineage from: {args.ground_truth}")

    predicted_data = load_json_file(args.predicted)
    ground_truth_data = load_json_file(args.ground_truth)

    # Parse custom weights if provided
    weights = None
    if args.weights:
        try:
            weight_pairs = args.weights.split(",")
            weights = {}
            for pair in weight_pairs:
                key, value = pair.split("=")
                weights[key.strip()] = float(value.strip())

            # Validate weights sum to 1.0
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                print(f"Warning: Weights sum to {total:.3f}, not 1.0")

        except (ValueError, AttributeError) as e:
            print(f"Error parsing weights: {e}")
            print(
                "Expected format: source_table=0.4,transformation=0.4,aggregation=0.2"
            )
            sys.exit(1)

    # Initialize evaluator
    evaluator = SchemaLineageEvaluator(weights=weights, has_metadata=args.metadata)

    # Evaluate
    if args.verbose:
        print("Evaluating lineage...")

    try:
        results = evaluator.evaluate(predicted_data, ground_truth_data)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

    # Format and output results
    formatted_results = format_results(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(formatted_results)
        if args.verbose:
            print(f"Results written to: {args.output}")
    else:
        print(formatted_results)


if __name__ == "__main__":
    main()
