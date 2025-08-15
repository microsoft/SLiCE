#!/usr/bin/env python3
"""
SLiCE Sample Data Usage Demonstration

This script demonstrates how to use the SLiCE sample data module to access
pipeline scripts, lineage information, and prepare data for batch evaluation.
"""

from slice.data import (
    list_inventory,
    get_pipeline_script,
    get_lineages,
    prepare_batch_evaluation,
    list_pipeline_scripts
)
from slice.SchemaLineageEvaluator import SchemaLineageEvaluator


def main():
    """
    Demonstrate the usage of SLiCE sample data functions.
    """
    print("=== SLiCE Sample Data Usage Demonstration ===")
    print("Exploring available sample datasets and their usage.\n")
    
    # 1. List available datasets
    print("1. DATASET INVENTORY")
    print("="*50)
    inventory = list_inventory()
    print(f"Total datasets available: {len(inventory)}")
    print("\nDataset overview:")
    for idx, row in inventory.iterrows():
        print(f"  ID: {row['id']}")
        print(f"  Language: {row['language']}")
        print(f"  Difficulty: {row['difficulty']}")
        print(f"  Columns: {row['column_count']}")
        print(f"  Last Modified: {row['last_modified_date']}")
        print("-" * 40)
    
    # 2. List available pipeline scripts
    print("\n2. PIPELINE SCRIPTS")
    print("="*50)
    scripts = list_pipeline_scripts()
    print(f"Total pipeline scripts available: {len(scripts)}")
    print("\nPipeline script overview:")
    for script in scripts:
        print(f"  ID: {script['id']}")
        print(f"  Type: {script['label_type']}")
        print(f"  Final Table: {script['final_table']}")
        print(f"  Difficulty: {script['difficulty']}")
        print("-" * 40)
    
    # 3. Get a specific pipeline script
    print("\n3. PIPELINE SCRIPT EXAMPLE")
    print("="*50)
    # Use the first dataset ID from inventory
    example_id = inventory['id'].iloc[0]
    pipeline_script = get_pipeline_script(example_id)
    
    print(f"Pipeline script for dataset {example_id}:")
    print(f"Script length: {len(pipeline_script)} characters")
    print(f"First 200 characters:\n{pipeline_script[:200]}...")
    print(f"Last 200 characters:\n...{pipeline_script[-200:]}")
    
    # 4. Get lineage information
    print("\n4. LINEAGE INFORMATION")
    print("="*50)
    
    # Get gold (human labeled) lineages
    print(f"Getting gold lineages for dataset {example_id}:")
    gold_lineages = get_lineages(id=example_id, label_type='gold')
    print(f"Number of schema columns: {len(gold_lineages)}")
    print("\nAvailable schema columns:")
    for column_name in gold_lineages.keys():
        print(f"  - {column_name}")
    
    # Show detailed lineage for one column
    first_column = list(gold_lineages.keys())[0]
    print(f"\nDetailed lineage for '{first_column}':")
    lineage_info = gold_lineages[first_column]
    for key, value in lineage_info.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")
    
    # Get generated lineages
    print(f"\nGetting generated lineages for dataset {example_id}:")
    try:
        generated_lineages = get_lineages(id=example_id, label_type='generated')
        print(f"Number of schema columns: {len(generated_lineages)}")
        print("Available schema columns:")
        for column_name in generated_lineages.keys():
            print(f"  - {column_name}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # 5. Get lineage for a specific column
    print("\n5. SPECIFIC COLUMN LINEAGE")
    print("="*50)
    specific_column_lineage = get_lineages(id=example_id, label_type='gold', columns=first_column)
    print(f"Lineage for specific column '{first_column}':")
    for key, value in specific_column_lineage[first_column].items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")
    
    # 6. Prepare batch evaluation data
    print("\n6. BATCH EVALUATION PREPARATION")
    print("="*50)
    
    # Try to prepare batch evaluation data if both gold and generated exist
    try:
        generated_lineages = get_lineages(id=example_id, label_type='generated')
        
        print("Preparing batch evaluation data...")
        gold_ordered, generated_ordered = prepare_batch_evaluation(gold_lineages, generated_lineages)
        
        print(f"Gold ordered data: {len(gold_ordered)} items")
        print(f"Generated ordered data: {len(generated_ordered)} items")
        
        print("\nSchema matching verification:")
        for i, (gold_item, generated_item) in enumerate(zip(gold_ordered, generated_ordered)):
            gold_schema, gold_data = gold_item
            generated_schema, generated_data = generated_item
            print(f"  {i+1}. {gold_schema} == {generated_schema}: {gold_schema == generated_schema}")
        
        # 7. Batch evaluation example
        print("\n7. BATCH EVALUATION EXAMPLE")
        print("="*50)
        
        if gold_ordered and generated_ordered:
            evaluator = SchemaLineageEvaluator()
            
            print(f"Performing batch evaluation on {len(gold_ordered)} schema columns...")
            
            # Batch evaluation results
            batch_results = []
            total_scores = {
                'overall': 0,
                'source_schema': 0,
                'source_table': 0,
                'transformation': 0,
                'aggregation': 0
            }
            
            print("\nIndividual schema evaluations:")
            print("-" * 70)
            print(f"{'Schema Name':<25} {'Overall':<8} {'SrcSchema':<10} {'SrcTable':<8} {'Transform':<9} {'Aggreg':<8}")
            print("-" * 70)
            
            # Evaluate each schema in order
            for i, (gold_item, generated_item) in enumerate(zip(gold_ordered, generated_ordered)):
                gold_schema, gold_data = gold_item
                generated_schema, generated_data = generated_item
                
                # Ensure schemas match (should always be true due to prepare_batch_evaluation)
                assert gold_schema == generated_schema, f"Schema mismatch: {gold_schema} != {generated_schema}"
                
                # Evaluate this schema
                results = evaluator.evaluate(generated_data, gold_data)
                
                # Store results
                batch_results.append({
                    'schema': gold_schema,
                    'results': results
                })
                
                # Add to totals
                for metric in total_scores:
                    total_scores[metric] += results.get(metric, 0)
                
                # Display individual results
                print(f"{gold_schema:<25} "
                      f"{results.get('overall', 0):<8.4f} "
                      f"{results.get('source_schema', 0):<10.4f} "
                      f"{results.get('source_table', 0):<8.4f} "
                      f"{results.get('transformation', 0):<9.4f} "
                      f"{results.get('aggregation', 0):<8.4f}")
            
            print("-" * 70)
            
            # Calculate and display average scores
            num_schemas = len(batch_results)
            avg_scores = {metric: total / num_schemas for metric, total in total_scores.items()}
            
            print(f"{'AVERAGE':<25} "
                  f"{avg_scores['overall']:<8.4f} "
                  f"{avg_scores['source_schema']:<10.4f} "
                  f"{avg_scores['source_table']:<8.4f} "
                  f"{avg_scores['transformation']:<9.4f} "
                  f"{avg_scores['aggregation']:<8.4f}")
            
            print("\n" + "="*50)
            print("BATCH EVALUATION SUMMARY")
            print("="*50)
            print(f"Total schemas evaluated: {num_schemas}")
            print(f"Average overall score: {avg_scores['overall']:.4f}")
            
            # Find best and worst performing schemas
            best_schema = max(batch_results, key=lambda x: x['results'].get('overall', 0))
            worst_schema = min(batch_results, key=lambda x: x['results'].get('overall', 0))
            
            print(f"\nBest performing schema: {best_schema['schema']}")
            print(f"  Score: {best_schema['results'].get('overall', 0):.4f}")
            
            print(f"\nWorst performing schema: {worst_schema['schema']}")
            print(f"  Score: {worst_schema['results'].get('overall', 0):.4f}")
            
            # Component analysis
            print("\nComponent-wise average scores:")
            print(f"  Source Schema: {avg_scores['source_schema']:.4f}")
            print(f"  Source Table:  {avg_scores['source_table']:.4f}")
            print(f"  Transformation: {avg_scores['transformation']:.4f}")
            print(f"  Aggregation:   {avg_scores['aggregation']:.4f}")
            
            # Performance distribution
            score_ranges = {
                'excellent': [0.9, 1.0],
                'good': [0.7, 0.9],
                'fair': [0.5, 0.7],
                'poor': [0.0, 0.5]
            }
            
            print("\nPerformance distribution:")
            for category, (min_score, max_score) in score_ranges.items():
                count = sum(1 for result in batch_results 
                           if min_score <= result['results'].get('overall', 0) < max_score)
                percentage = (count / num_schemas) * 100
                print(f"  {category.capitalize()}: {count}/{num_schemas} ({percentage:.1f}%)")
            
            # Store batch results for potential further analysis
            print(f"\nBatch evaluation complete! Results stored for {num_schemas} schemas.")
            print("This demonstrates how to:")
            print("- Process multiple schemas in order")
            print("- Calculate aggregate statistics")
            print("- Identify best/worst performing schemas")
            print("- Analyze component-wise performance")
        
    except ValueError as e:
        print(f"Cannot prepare batch evaluation: {e}")
        print("This likely means the generated data doesn't have corresponding lineages for this dataset.")
    
    # 8. Explore different datasets
    print("\n8. EXPLORING DIFFERENT DATASETS")
    print("="*50)
    
    print("Summary of all available datasets:")
    for _, row in inventory.iterrows():
        dataset_id = row['id']
        try:
            gold_lineages = get_lineages(id=dataset_id, label_type='gold')
            try:
                generated_lineages = get_lineages(id=dataset_id, label_type='generated')
                has_generated = True
            except ValueError:
                has_generated = False
            
            print(f"  Dataset: {dataset_id}")
            print(f"    Language: {row['language']}")
            print(f"    Difficulty: {row['difficulty']}")
            print(f"    Schema columns: {len(gold_lineages)}")
            print(f"    Has generated lineages: {has_generated}")
            print("-" * 40)
            
        except ValueError:
            print(f"  Dataset: {dataset_id} - ERROR: Cannot load lineages")
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETE")
    print("="*50)
    print("This example shows how to:")
    print("1. Access dataset inventory and metadata")
    print("2. Retrieve pipeline scripts")
    print("3. Load lineage information (gold and generated)")
    print("4. Prepare data for batch evaluation")
    print("5. Perform individual schema lineage evaluations")
    print("\nUse these functions to build your own evaluation workflows!")


if __name__ == "__main__":
    main()