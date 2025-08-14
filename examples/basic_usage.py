#!/usr/bin/env python3
"""
SLiCE: Schema Lineage Calculation and Evaluation
Main demonstration script for calculating schema lineage accuracy using default settings.

This script demonstrates how to use the SchemaLineageEvaluator to calculate accuracy
scores between predicted and ground truth schema lineage data.
"""

from slice.SchemaLineageEvaluator import SchemaLineageEvaluator

def main():
    """
    Demonstrate schema lineage accuracy calculation using default settings.
    """
    print("=== SLiCE: Schema Lineage Calculation and Evaluation ===")
    print("Demonstrating schema lineage accuracy calculation with default settings.\n")
    
    # Initialize evaluator with default settings
    evaluator = SchemaLineageEvaluator()
    
    # Example predicted lineage (from the user's example)
    predicted = {
        "source_schema": "cuisine_type",
        "source_table": "RestaurantsPath = string.Format(@\"/shares/local/FoodDelivery/Restaurants/restaurants{0}.ss\",@startDate); #DECLARE startDate DateTime = DateTime.Today.AddDays(-1);",
        "transformation": "",
        "aggregation": "COUNT() GROUP BY BD.restaurant_id <CODEEND> SUM() GROUP BY BD.restaurant_id"
    }
    
    # Ground truth lineage (from the user's example)
    truth = {
        "source_schema": "cuisine_type", 
        "source_table": "#DECLARE RestaurantsPath string = string.Format(@\"/shares/local/FoodDelivery/Restaurants/restaurants{0}.ss\",@startDate); #DECLARE startDate DateTime = DateTime.Today.AddDays(-1);",
        "transformation": "R.cuisine_type AS CuisineType",
        "aggregation": ""
    }
    
    print("Predicted lineage:")
    for key, value in predicted.items():
        print(f"  {key}: {value}")
    
    print("\nGround truth lineage:")
    for key, value in truth.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Evaluate the lineage
    results = evaluator.evaluate(predicted, truth)
    
    # Display results
    print(f"Overall Score: {results.get('overall', 0):.4f}")
    print(f"Format Correctness: {results.get('format', 0):.4f}")
    print(f"Source Schema Score: {results.get('source_schema', 0):.4f}")
    print(f"Source Table Score: {results.get('source_table', 0):.4f}")
    print(f"Transformation Score: {results.get('transformation', 0):.4f}")
    print(f"Aggregation Score: {results.get('aggregation', 0):.4f}")
    
    print("\n" + "="*50)
    print("COMPONENT ANALYSIS")
    print("="*50)
    
    # Detailed analysis
    print("\n1. Source Schema Analysis:")
    print(f"   - Both predicted and truth have: '{predicted['source_schema']}'")
    print(f"   - Exact match: {'' if results.get('source_schema', 0) == 1.0 else ''}")
    
    print("\n2. Source Table Analysis:")
    print(f"   - Score: {results.get('source_table', 0):.4f}")
    print("   - Uses fuzzy matching and F1 score combination")
    
    print("\n3. Transformation Analysis:")
    print(f"   - Score: {results.get('transformation', 0):.4f}")
    print("   - Predicted: Empty string")
    print("   - Truth: 'R.cuisine_type AS CuisineType'")
    print("   - Uses BLEU, weighted BLEU, and AST similarity")
    
    print("\n4. Aggregation Analysis:")
    print(f"   - Score: {results.get('aggregation', 0):.4f}")
    print("   - Predicted: 'COUNT() GROUP BY BD.restaurant_id <CODEEND> SUM() GROUP BY BD.restaurant_id'")
    print("   - Truth: Empty string")
    print("   - Uses BLEU, weighted BLEU, and AST similarity")
    
    print("\n" + "="*50)
    print("DEFAULT CONFIGURATION")
    print("="*50)
    print("Weights used for final score calculation:")
    print(f"  - Source Table: {evaluator.weights.get('source_table', 0.4)}")
    print(f"  - Transformation: {evaluator.weights.get('transformation', 0.4)}")
    print(f"  - Aggregation: {evaluator.weights.get('aggregation', 0.2)}")
    
    print("\nComponent-specific weights:")
    print("  Source Table weights:", evaluator.source_table_weights)
    print("  Transformation weights:", evaluator.transformation_weights)
    print("  Aggregation weights:", evaluator.aggregation_weights)
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETE")
    print("="*50)
    print("This example shows how SLiCE evaluates schema lineage accuracy")
    print("by comparing predicted and ground truth lineage components.")

if __name__ == "__main__":
    main()