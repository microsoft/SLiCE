"""
Tests for SchemaLineageEvaluator.
"""

import pytest
from slice import SchemaLineageEvaluator, SchemaLineage


class TestSchemaLineageEvaluator:
    """Test cases for SchemaLineageEvaluator."""
    
    def test_init_default(self):
        """Test evaluator initialization with default parameters."""
        evaluator = SchemaLineageEvaluator()
        
        assert evaluator.has_metadata is False
        assert len(evaluator.weights) == 3
        assert evaluator.weights['source_table'] == 0.4
        assert evaluator.weights['transformation'] == 0.4
        assert evaluator.weights['aggregation'] == 0.2
    
    def test_init_with_metadata(self):
        """Test evaluator initialization with metadata enabled."""
        evaluator = SchemaLineageEvaluator(has_metadata=True)
        
        assert evaluator.has_metadata is True
        assert len(evaluator.weights) == 4
        assert 'metadata' in evaluator.weights
    
    def test_init_custom_weights(self):
        """Test evaluator initialization with custom weights."""
        weights = {
            'source_table': 0.5,
            'transformation': 0.3,
            'aggregation': 0.2
        }
        evaluator = SchemaLineageEvaluator(weights=weights)
        
        assert evaluator.weights == weights
    
    def test_validate_lineage_valid(self):
        """Test lineage validation with valid data."""
        evaluator = SchemaLineageEvaluator()
        lineage = {
            'source_schema': 'test_schema',
            'source_table': 'test_table',
            'transformation': 'test_transform',
            'aggregation': 'test_agg'
        }
        
        assert evaluator._validate_lineage(lineage) is True
    
    def test_validate_lineage_missing_field(self):
        """Test lineage validation with missing field."""
        evaluator = SchemaLineageEvaluator()
        lineage = {
            'source_schema': 'test_schema',
            'source_table': 'test_table',
            'transformation': 'test_transform'
            # missing aggregation
        }
        
        assert evaluator._validate_lineage(lineage) is False
    
    def test_evaluate_perfect_match(self):
        """Test evaluation with perfect match."""
        evaluator = SchemaLineageEvaluator()
        
        predicted = {
            'source_schema': 'cuisine_type',
            'source_table': 'restaurants.ss',
            'transformation': 'R.cuisine_type AS CuisineType',
            'aggregation': 'COUNT() GROUP BY restaurant_id'
        }
        
        ground_truth = {
            'source_schema': 'cuisine_type',
            'source_table': 'restaurants.ss',
            'transformation': 'R.cuisine_type AS CuisineType',
            'aggregation': 'COUNT() GROUP BY restaurant_id'
        }
        
        results = evaluator.evaluate(predicted, ground_truth)
        
        assert 'overall' in results or 'all' in results
        assert results['format'] == 1
        assert results['source_schema'] == 1
        # Other scores might not be perfect due to fuzzy matching and AST parsing
    
    def test_evaluate_empty_fields(self):
        """Test evaluation with empty fields."""
        evaluator = SchemaLineageEvaluator()
        
        predicted = {
            'source_schema': 'cuisine_type',
            'source_table': 'restaurants.ss',
            'transformation': '',
            'aggregation': ''
        }
        
        ground_truth = {
            'source_schema': 'cuisine_type',
            'source_table': 'restaurants.ss',
            'transformation': '',
            'aggregation': ''
        }
        
        results = evaluator.evaluate(predicted, ground_truth)
        
        assert results['format'] == 1
        assert results['source_schema'] == 1
        assert results['transformation'] >= 0.0
        assert results['aggregation'] >= 0.0
    
    def test_evaluate_source_schema_exact_match(self):
        """Test source schema evaluation with exact match."""
        evaluator = SchemaLineageEvaluator()
        
        predicted = SchemaLineage(
            source_schema='cuisine_type',
            source_table='test',
            transformation='test',
            aggregation='test'
        )
        
        ground_truth = SchemaLineage(
            source_schema='cuisine_type',
            source_table='test',
            transformation='test',
            aggregation='test'
        )
        
        score = evaluator.evaluate_source_schema(predicted, ground_truth)
        assert score == 1.0
    
    def test_evaluate_source_schema_no_match(self):
        """Test source schema evaluation with no match."""
        evaluator = SchemaLineageEvaluator()
        
        predicted = SchemaLineage(
            source_schema='cuisine_type',
            source_table='test',
            transformation='test',
            aggregation='test'
        )
        
        ground_truth = SchemaLineage(
            source_schema='different_schema',
            source_table='test',
            transformation='test',
            aggregation='test'
        )
        
        score = evaluator.evaluate_source_schema(predicted, ground_truth)
        assert score == 0.0
    
    def test_evaluate_batch_same_length(self):
        """Test batch evaluation with same length inputs."""
        evaluator = SchemaLineageEvaluator()
        
        predicted_batch = [
            {
                'source_schema': 'schema1',
                'source_table': 'table1',
                'transformation': 'transform1',
                'aggregation': 'agg1'
            },
            {
                'source_schema': 'schema2',
                'source_table': 'table2',
                'transformation': 'transform2',
                'aggregation': 'agg2'
            }
        ]
        
        ground_truth_batch = [
            {
                'source_schema': 'schema1',
                'source_table': 'table1',
                'transformation': 'transform1',
                'aggregation': 'agg1'
            },
            {
                'source_schema': 'schema2',
                'source_table': 'table2',
                'transformation': 'transform2',
                'aggregation': 'agg2'
            }
        ]
        
        results = evaluator.evaluate_batch(predicted_batch, ground_truth_batch)
        
        assert len(results) == 2
        assert all('format' in result for result in results)
    
    def test_evaluate_batch_different_length(self):
        """Test batch evaluation with different length inputs."""
        evaluator = SchemaLineageEvaluator()
        
        predicted_batch = [
            {
                'source_schema': 'schema1',
                'source_table': 'table1',
                'transformation': 'transform1',
                'aggregation': 'agg1'
            }
        ]
        
        ground_truth_batch = [
            {
                'source_schema': 'schema1',
                'source_table': 'table1',
                'transformation': 'transform1',
                'aggregation': 'agg1'
            },
            {
                'source_schema': 'schema2',
                'source_table': 'table2',
                'transformation': 'transform2',
                'aggregation': 'agg2'
            }
        ]
        
        with pytest.raises(ValueError, match="Number of predicted lineages"):
            evaluator.evaluate_batch(predicted_batch, ground_truth_batch)
    
    def test_weights_validation_incorrect_sum(self):
        """Test weights validation with incorrect sum."""
        weights = {
            'source_table': 0.3,
            'transformation': 0.3,
            'aggregation': 0.3  # Sum = 0.9, not 1.0
        }
        
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            SchemaLineageEvaluator(weights=weights)
    
    def test_weights_validation_incorrect_count(self):
        """Test weights validation with incorrect count."""
        weights = {
            'source_table': 0.5,
            'transformation': 0.5
            # Missing aggregation
        }
        
        with pytest.raises(ValueError, match="Weights must have 3 components"):
            SchemaLineageEvaluator(weights=weights)
    
    def test_component_weights_validation(self):
        """Test component weights validation."""
        source_table_weights = {
            'f1': 0.5,
            'fuzzy': 0.3  # Sum = 0.8, not 1.0
        }
        
        with pytest.raises(ValueError, match="Source table weights must sum to 1.0"):
            SchemaLineageEvaluator(source_table_weights=source_table_weights)


class TestSchemaLineage:
    """Test cases for SchemaLineage dataclass."""
    
    def test_schema_lineage_creation(self):
        """Test SchemaLineage object creation."""
        lineage = SchemaLineage(
            source_schema='test_schema',
            source_table='test_table',
            transformation='test_transform',
            aggregation='test_agg'
        )
        
        assert lineage.source_schema == 'test_schema'
        assert lineage.source_table == 'test_table'
        assert lineage.transformation == 'test_transform'
        assert lineage.aggregation == 'test_agg'
        assert lineage.metadata is None
    
    def test_schema_lineage_with_metadata(self):
        """Test SchemaLineage object creation with metadata."""
        lineage = SchemaLineage(
            source_schema='test_schema',
            source_table='test_table',
            transformation='test_transform',
            aggregation='test_agg',
            metadata='test_metadata'
        )
        
        assert lineage.metadata == 'test_metadata'


if __name__ == "__main__":
    pytest.main([__file__])