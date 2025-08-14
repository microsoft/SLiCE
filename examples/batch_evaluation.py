#!/usr/bin/env python3
"""
SLiCE: Batch Evaluation Example
Example showing how to evaluate multiple lineage pairs efficiently.
"""

from slice import SchemaLineageEvaluator
import time


def main():
    """
    Demonstrate batch evaluation of multiple schema lineage pairs.
    """
    print("=== SLiCE: Batch Evaluation Example ===")
    print("Demonstrating batch evaluation for multiple lineage pairs.\n")
    
    # Initialize evaluator
    evaluator = SchemaLineageEvaluator()
    
    # Create multiple test cases
    predicted_batch = [
        {
            "source_schema": "cuisine_type",
            "source_table": "restaurants.ss",
            "transformation": "R.cuisine_type AS CuisineType",
            "aggregation": "COUNT() GROUP BY restaurant_id"
        },
        {
            "source_schema": "order_amount",
            "source_table": "orders.csv",
            "transformation": "O.amount * 1.1 AS TotalAmount",
            "aggregation": "SUM(amount) GROUP BY customer_id"
        },
        {
            "source_schema": "customer_name",
            "source_table": "customers.db",
            "transformation": "CONCAT(first_name, ' ', last_name) AS FullName",
            "aggregation": ""
        },
        {
            "source_schema": "product_category",
            "source_table": "products.json",
            "transformation": "P.category AS ProductCategory",
            "aggregation": "COUNT(DISTINCT category)"
        },
        {
            "source_schema": "delivery_time",
            "source_table": "deliveries.ss",
            "transformation": "DATEDIFF(delivered_at, ordered_at) AS DeliveryDays",
            "aggregation": "AVG(delivery_time) GROUP BY region"
        }
    ]
    
    ground_truth_batch = [
        {
            "source_schema": "cuisine_type",
            "source_table": "restaurants.ss",
            "transformation": "R.cuisine_type AS CuisineType",
            "aggregation": "COUNT() GROUP BY restaurant_id"
        },
        {
            "source_schema": "order_amount",
            "source_table": "orders.csv",
            "transformation": "O.amount AS TotalAmount",  # Different transformation
            "aggregation": "SUM(amount) GROUP BY customer_id"
        },
        {
            "source_schema": "customer_name",
            "source_table": "customers.database",  # Different table name
            "transformation": "CONCAT(first_name, ' ', last_name) AS FullName",
            "aggregation": ""
        },
        {
            "source_schema": "product_category",
            "source_table": "products.json",
            "transformation": "P.category AS ProductCategory",
            "aggregation": "COUNT(DISTINCT category)"
        },
        {
            "source_schema": "delivery_time",
            "source_table": "deliveries.ss",
            "transformation": "DATEDIFF(delivered_at, ordered_at) AS DeliveryDays",
            "aggregation": "AVG(delivery_time)"  # Different aggregation
        }
    ]
    
    print(f"Evaluating {len(predicted_batch)} lineage pairs...\n")
    
    # Measure time for batch evaluation
    start_time = time.time()
    batch_results = evaluator.evaluate_batch(predicted_batch, ground_truth_batch)
    batch_time = time.time() - start_time
    
    print("=" * 60)
    print("BATCH EVALUATION RESULTS")
    print("=" * 60)
    
    # Display results for each pair
    for i, results in enumerate(batch_results, 1):
        print(f"\nLineage Pair {i}:")
        print(f"  Overall Score: {results.get('overall', results.get('all', 0)):.4f}")
        print(f"  Format: {results.get('format', 0):.4f}")
        print(f"  Source Schema: {results.get('source_schema', 0):.4f}")
        print(f"  Source Table: {results.get('source_table', 0):.4f}")
        print(f"  Transformation: {results.get('transformation', 0):.4f}")
        print(f"  Aggregation: {results.get('aggregation', 0):.4f}")
    
    # Calculate summary statistics
    overall_scores = [r.get('overall', r.get('all', 0)) for r in batch_results]
    avg_score = sum(overall_scores) / len(overall_scores)
    max_score = max(overall_scores)
    min_score = min(overall_scores)
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Average Overall Score: {avg_score:.4f}")
    print(f"Maximum Overall Score: {max_score:.4f}")
    print(f"Minimum Overall Score: {min_score:.4f}")
    print(f"Batch Processing Time: {batch_time:.3f} seconds")
    print(f"Average Time per Pair: {batch_time / len(predicted_batch):.3f} seconds")
    
    # Compare with individual evaluation
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    start_time = time.time()
    individual_results = []
    for pred, truth in zip(predicted_batch, ground_truth_batch):
        result = evaluator.evaluate(pred, truth)
        individual_results.append(result)
    individual_time = time.time() - start_time
    
    print(f"Individual Processing Time: {individual_time:.3f} seconds")
    print(f"Batch Processing Time: {batch_time:.3f} seconds")
    
    if batch_time < individual_time:
        speedup = individual_time / batch_time
        print(f"Batch processing is {speedup:.2f}x faster!")
    else:
        print("Individual processing was faster (small batch size)")
    
    # Verify results are the same
    results_match = True
    for batch_result, individual_result in zip(batch_results, individual_results):
        for key in batch_result:
            if abs(batch_result[key] - individual_result[key]) > 1e-10:
                results_match = False
                break
        if not results_match:
            break
    
    print(f"Results consistency: {'✓ PASS' if results_match else '✗ FAIL'}")
    
    # Component-wise analysis
    print("\n" + "=" * 60)
    print("COMPONENT-WISE ANALYSIS")
    print("=" * 60)
    
    components = ['source_schema', 'source_table', 'transformation', 'aggregation']
    for component in components:
        scores = [r.get(component, 0) for r in batch_results]
        avg_score = sum(scores) / len(scores)
        print(f"{component.replace('_', ' ').title()}: {avg_score:.4f} average")
    
    # Identify best and worst performing pairs
    print("\n" + "=" * 60)
    print("PERFORMANCE HIGHLIGHTS")
    print("=" * 60)
    
    best_idx = overall_scores.index(max_score)
    worst_idx = overall_scores.index(min_score)
    
    print(f"Best performing pair: #{best_idx + 1} (Score: {max_score:.4f})")
    print(f"  Schema: {predicted_batch[best_idx]['source_schema']}")
    print(f"  Perfect matches likely in multiple components")
    
    print(f"\nWorst performing pair: #{worst_idx + 1} (Score: {min_score:.4f})")
    print(f"  Schema: {predicted_batch[worst_idx]['source_schema']}")
    print(f"  Check for differences in transformations or aggregations")
    
    print("\n" + "=" * 60)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 60)
    print("This example demonstrates:")
    print("- Efficient batch processing of multiple lineage pairs")
    print("- Performance comparison between batch and individual evaluation")
    print("- Summary statistics and component-wise analysis")
    print("- Identification of best and worst performing pairs")


if __name__ == "__main__":
    main()