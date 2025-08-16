#!/usr/bin/env python3
"""
SLiCE: Custom Weights Example
Example showing how to use custom weights and configurations.
"""

from slice import SchemaLineageEvaluator


def main():
    """
    Demonstrate schema lineage evaluation with custom weights and configurations.
    """
    print("=== SLiCE: Custom Weights Example ===")
    print("Demonstrating custom weights and configurations.\n")

    # Custom component weights (must sum to 1.0)
    component_weights = {
        "source_table": 0.5,  # Higher weight for source table accuracy
        "transformation": 0.3,
        "aggregation": 0.2,
    }

    # Custom metric weights for transformations
    transformation_weights = {
        "bleu": 0.6,  # Higher weight for BLEU score
        "weighted_bleu": 0.3,
        "ast": 0.1,
    }

    # Custom metric weights for aggregations
    aggregation_weights = {
        "bleu": 0.4,
        "weighted_bleu": 0.4,
        "ast": 0.2,  # Higher weight for AST similarity
    }

    # Initialize evaluator with custom settings
    evaluator = SchemaLineageEvaluator(
        weights=component_weights,
        transformation_weights=transformation_weights,
        aggregation_weights=aggregation_weights,
    )

    # Example data with some differences
    predicted = {
        "source_schema": "cuisine_type",
        "source_table": "restaurants_data.csv",  # Different table name
        "transformation": "SELECT cuisine_type AS CuisineType",  # Different syntax
        "aggregation": "COUNT(*) GROUP BY restaurant_id",
    }

    ground_truth = {
        "source_schema": "cuisine_type",
        "source_table": "restaurants.ss",
        "transformation": "R.cuisine_type AS CuisineType",
        "aggregation": "COUNT() GROUP BY restaurant_id",
    }

    print("Custom Configuration:")
    print(f"Component weights: {component_weights}")
    print(f"Transformation weights: {transformation_weights}")
    print(f"Aggregation weights: {aggregation_weights}")

    print("\nPredicted lineage:")
    for key, value in predicted.items():
        print(f"  {key}: {value}")

    print("\nGround truth lineage:")
    for key, value in ground_truth.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS (Custom Weights)")
    print("=" * 50)

    # Evaluate with custom configuration
    results = evaluator.evaluate(predicted, ground_truth)

    # Display results
    print(f"Overall Score: {results.get('overall', results.get('all', 0)):.4f}")
    print(f"Format Correctness: {results.get('format', 0):.4f}")
    print(f"Source Schema Score: {results.get('source_schema', 0):.4f}")
    print(f"Source Table Score: {results.get('source_table', 0):.4f}")
    print(f"Transformation Score: {results.get('transformation', 0):.4f}")
    print(f"Aggregation Score: {results.get('aggregation', 0):.4f}")

    print("\n" + "=" * 50)
    print("COMPARISON WITH DEFAULT WEIGHTS")
    print("=" * 50)

    # Compare with default weights
    default_evaluator = SchemaLineageEvaluator()
    default_results = default_evaluator.evaluate(predicted, ground_truth)

    print(f"Custom Overall Score:  {results.get('overall', results.get('all', 0)):.4f}")
    print(
        f"Default Overall Score: {default_results.get('overall', default_results.get('all', 0)):.4f}"
    )

    # Calculate the difference
    custom_overall = results.get("overall", results.get("all", 0))
    default_overall = default_results.get("overall", default_results.get("all", 0))
    difference = custom_overall - default_overall

    print(f"Difference: {difference:+.4f}")

    if difference > 0:
        print("Custom weights resulted in a higher score!")
    elif difference < 0:
        print("Default weights resulted in a higher score!")
    else:
        print("Both configurations yielded the same score.")

    print("\n" + "=" * 50)
    print("ADVANCED CONFIGURATION EXAMPLE")
    print("=" * 50)

    # Example with metadata enabled
    print("\nTesting with metadata enabled...")

    metadata_weights = {
        "source_table": 0.3,
        "transformation": 0.3,
        "aggregation": 0.2,
        "metadata": 0.2,
    }

    # Add metadata to the data
    predicted_with_metadata = predicted.copy()
    predicted_with_metadata["metadata"] = "Pipeline metadata info"

    ground_truth_with_metadata = ground_truth.copy()
    ground_truth_with_metadata["metadata"] = "Reference metadata info"

    metadata_evaluator = SchemaLineageEvaluator(
        weights=metadata_weights, has_metadata=True
    )

    metadata_results = metadata_evaluator.evaluate(
        predicted_with_metadata, ground_truth_with_metadata
    )

    print(
        f"Metadata-enabled Overall Score: {metadata_results.get('overall', metadata_results.get('all', 0)):.4f}"
    )
    print(f"Metadata Component Score: {metadata_results.get('metadata', 0):.4f}")

    print("\n" + "=" * 50)
    print("CUSTOM CONFIGURATION COMPLETE")
    print("=" * 50)
    print("This example shows how to:")
    print("- Use custom component weights")
    print("- Configure metric-specific weights")
    print("- Compare results with different configurations")
    print("- Enable metadata evaluation")


if __name__ == "__main__":
    main()
