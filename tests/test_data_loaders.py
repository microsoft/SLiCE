"""
Tests for slice.data.loaders module.
"""

import pytest
import pandas as pd
from pathlib import Path
import json
from slice.data.loaders import (
    list_inventory, 
    get_pipeline_script, 
    get_lineages, 
    prepare_batch_evaluation,
    list_pipeline_scripts,
    GOLD,
    GENERATED,
    INVENTORY_COLUMNS
)


class TestDataExistence:
    """Test that all required data files exist and have correct structure."""
    
    def test_index_csv_exists(self):
        """Test that index.csv exists and has expected columns."""
        data_path = Path(__file__).parent.parent / "slice" / "data" / "datasets"
        index_file = data_path / "index.csv"
        assert index_file.exists(), "index.csv file not found"
        
        # Test that file can be read
        df = pd.read_csv(index_file)
        assert not df.empty, "index.csv is empty"
        
        # Test expected columns exist
        for col in INVENTORY_COLUMNS:
            assert col in df.columns, f"Expected column '{col}' not found in index.csv"
    
    def test_gold_json_exists(self):
        """Test that human_labeled.json exists and has valid structure."""
        data_path = Path(__file__).parent.parent / "slice" / "data" / "datasets"
        gold_file = data_path / GOLD
        assert gold_file.exists(), f"{GOLD} file not found"
        
        # Test that file can be loaded as JSON
        with open(gold_file, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, dict), "Gold data should be a dictionary"
        assert len(data) > 0, "Gold data should not be empty"
        
        # Test structure of at least one entry
        for dataset_id, dataset_info in data.items():
            if isinstance(dataset_info, dict):
                assert 'lineage' in dataset_info, f"Dataset {dataset_id} missing 'lineage' key"
                assert 'pipeline_script' in dataset_info, f"Dataset {dataset_id} missing 'pipeline_script' key"
                break

    def test_id_match(self):
        inventory = list_inventory()
        inventory_ids = set(inventory['id'].tolist())

        gold_file = Path(__file__).parent.parent / "slice" / "data" / "datasets" / GOLD
        with open(gold_file, 'r') as f:
            gold_data = json.load(f)
        gold_lineages = set(gold_data.keys())

        generated_file = Path(__file__).parent.parent / "slice" / "data" / "datasets" / GENERATED
        with open(generated_file, 'r') as f:
            generated_data = json.load(f)
        generated_lineages = set(generated_data.keys())
        generated_lineages.remove('_metadata')

        assert inventory_ids == gold_lineages, "Inventory IDs do not match gold lineages"
        assert inventory_ids == generated_lineages, "Inventory IDs do not match generated lineages"

    def test_generated_json_exists(self):
        """Test that generated.json exists and has valid structure."""
        data_path = Path(__file__).parent.parent / "slice" / "data" / "datasets"
        generated_file = data_path / GENERATED
        assert generated_file.exists(), f"{GENERATED} file not found"
        
        # Test that file can be loaded as JSON
        with open(generated_file, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, dict), "Generated data should be a dictionary"
        assert len(data) > 0, "Generated data should not be empty"
        
        # Test structure of at least one entry
        for dataset_id, dataset_info in data.items():
            if isinstance(dataset_info, dict):
                assert 'parsed_lineages' in dataset_info, f"Dataset {dataset_id} missing 'parsed_lineages' key"
                # Note: pipeline_script is NOT in generated data
                break


class TestListInventory:
    """Test the list_inventory function."""
    
    def test_list_inventory_returns_dataframe(self):
        """Test that list_inventory returns a pandas DataFrame."""
        result = list_inventory()
        assert isinstance(result, pd.DataFrame), "list_inventory should return a DataFrame"
    
    def test_list_inventory_has_expected_columns(self):
        """Test that the returned DataFrame has expected columns."""
        result = list_inventory()
        for col in INVENTORY_COLUMNS:
            assert col in result.columns, f"Expected column '{col}' not found"
    
    def test_list_inventory_not_empty(self):
        """Test that the inventory is not empty."""
        result = list_inventory()
        assert len(result) > 0, "Inventory should not be empty"
    
    def test_list_inventory_has_valid_ids(self):
        """Test that inventory has valid dataset IDs."""
        result = list_inventory()
        assert 'id' in result.columns
        assert result['id'].notna().all(), "All IDs should be non-null"
        assert result['id'].str.len().gt(0).all(), "All IDs should be non-empty strings"


class TestGetPipelineScript:
    """Test the get_pipeline_script function."""
    
    def test_get_pipeline_script_default_params(self):
        """Test get_pipeline_script with default parameters."""
        script = get_pipeline_script()
        assert isinstance(script, str), "Pipeline script should be a string"
        assert len(script) > 0, "Pipeline script should not be empty"
    
    def test_get_pipeline_script_with_valid_id(self):
        """Test get_pipeline_script with a valid ID."""
        # Get first available ID from inventory
        inventory = list_inventory()
        test_id = inventory['id'].iloc[0]
        
        script = get_pipeline_script(id=test_id)
        assert isinstance(script, str), "Pipeline script should be a string"
    
    def test_get_pipeline_script_invalid_id(self):
        """Test get_pipeline_script with invalid ID."""
        with pytest.raises(ValueError, match="Dataset ID .* not found"):
            get_pipeline_script(id='invalid-id-12345')


class TestGetLineages:
    """Test the get_lineages function."""
    
    def test_get_lineages_default_params(self):
        """Test get_lineages with default parameters."""
        lineages = get_lineages()
        assert isinstance(lineages, dict), "Lineages should be a dictionary"
        assert len(lineages) > 0, "Lineages should not be empty"
    
    def test_get_lineages_gold_type(self):
        """Test get_lineages with gold label type."""
        lineages = get_lineages(label_type='gold')
        assert isinstance(lineages, dict), "Lineages should be a dictionary"
        
        # Test structure of lineage data
        for column_name, lineage_info in lineages.items():
            assert isinstance(lineage_info, dict), f"Lineage info for {column_name} should be dict"
            expected_keys = ['source_schema', 'source_table', 'transformation', 'aggregation']
            for key in expected_keys:
                assert key in lineage_info, f"Missing key '{key}' in lineage for {column_name}"
    
    def test_get_lineages_generated_type(self):
        """Test get_lineages with generated label type."""
        # Get first available ID from inventory
        inventory = list_inventory()
        test_id = inventory['id'].iloc[0]
        
        try:
            lineages = get_lineages(id=test_id, label_type='generated')
            assert isinstance(lineages, dict), "Lineages should be a dictionary"
        except ValueError:
            # It's ok if the ID doesn't exist in generated data
            pass
    
    def test_get_lineages_specific_column(self):
        """Test get_lineages with specific column filter."""
        # First get all lineages to find a valid column
        all_lineages = get_lineages()
        if len(all_lineages) > 0:
            column_name = list(all_lineages.keys())[0]
            
            # Now get just that column
            single_lineage = get_lineages(columns=column_name)
            assert len(single_lineage) == 1, "Should return only one column"
            assert column_name in single_lineage, f"Should contain column {column_name}"
    
    def test_get_lineages_invalid_column(self):
        """Test get_lineages with invalid column name."""
        with pytest.raises(ValueError, match="Column .* not found"):
            get_lineages(columns='invalid-column-name')
    
    def test_get_lineages_invalid_type(self):
        """Test get_lineages with invalid label type."""
        with pytest.raises(ValueError, match="Invalid label_type"):
            get_lineages(label_type='invalid')
    
    def test_get_lineages_invalid_id(self):
        """Test get_lineages with invalid ID."""
        with pytest.raises(ValueError, match="Dataset ID .* not found"):
            get_lineages(id='invalid-id-12345')


class TestPrepareBatchEvaluation:
    """Test the prepare_batch_evaluation function."""
    
    def test_prepare_batch_evaluation_matching_schemas(self):
        """Test prepare_batch_evaluation with matching schemas."""
        # Create mock data with matching schemas
        gold_data = {
            'column1': {'source_schema': 'col1', 'transformation': 'none'},
            'column2': {'source_schema': 'col2', 'transformation': 'upper'}
        }
        generated_data = {
            'column1': {'source_schema': 'col1_gen', 'transformation': 'none_gen'},
            'column2': {'source_schema': 'col2_gen', 'transformation': 'upper_gen'}
        }
        
        gold_ordered, generated_ordered = prepare_batch_evaluation(gold_data, generated_data)
        
        assert len(gold_ordered) == len(generated_ordered), "Should have same length"
        assert len(gold_ordered) == 2, "Should have 2 items each"
        
        # Check that schemas are in sorted order
        gold_schemas = [item[0] for item in gold_ordered]
        generated_schemas = [item[0] for item in generated_ordered]
        
        assert gold_schemas == generated_schemas, "Schema names should match"
        assert gold_schemas == sorted(gold_schemas), "Should be in sorted order"
    
    def test_prepare_batch_evaluation_mismatched_schemas(self):
        """Test prepare_batch_evaluation with mismatched schemas."""
        gold_data = {
            'column1': {'source_schema': 'col1'},
            'column2': {'source_schema': 'col2'}
        }
        generated_data = {
            'column1': {'source_schema': 'col1_gen'},
            'column3': {'source_schema': 'col3_gen'}  # Different column name
        }
        
        with pytest.raises(ValueError, match="Schema names mismatch"):
            prepare_batch_evaluation(gold_data, generated_data)
    
    def test_prepare_batch_evaluation_missing_in_generated(self):
        """Test prepare_batch_evaluation with missing columns in generated."""
        gold_data = {
            'column1': {'source_schema': 'col1'},
            'column2': {'source_schema': 'col2'}
        }
        generated_data = {
            'column1': {'source_schema': 'col1_gen'}
            # Missing column2
        }
        
        with pytest.raises(ValueError, match="Missing in generated"):
            prepare_batch_evaluation(gold_data, generated_data)
    
    def test_prepare_batch_evaluation_extra_in_generated(self):
        """Test prepare_batch_evaluation with extra columns in generated."""
        gold_data = {
            'column1': {'source_schema': 'col1'}
        }
        generated_data = {
            'column1': {'source_schema': 'col1_gen'},
            'column2': {'source_schema': 'col2_gen'}  # Extra column
        }
        
        with pytest.raises(ValueError, match="Extra in generated"):
            prepare_batch_evaluation(gold_data, generated_data)


class TestListPipelineScripts:
    """Test the list_pipeline_scripts function."""
    
    def test_list_pipeline_scripts_returns_list(self):
        """Test that list_pipeline_scripts returns a list."""
        scripts = list_pipeline_scripts()
        assert isinstance(scripts, list), "Should return a list"
    
    def test_list_pipeline_scripts_structure(self):
        """Test the structure of returned pipeline scripts."""
        scripts = list_pipeline_scripts()
        
        if len(scripts) > 0:
            script = scripts[0]
            expected_keys = ['id', 'label_type', 'final_table', 'difficulty']
            
            for key in expected_keys:
                assert key in script, f"Missing key '{key}' in script info"
            
            assert script['label_type'] in ['gold', 'generated'], "Invalid label_type"
           
    
    def test_list_pipeline_scripts_only_gold_type(self):
        """Test that list only includes gold scripts (pipeline scripts only in gold data)."""
        scripts = list_pipeline_scripts()
        
        if len(scripts) > 0:
            label_types = {script['label_type'] for script in scripts}
            assert label_types == {'gold'}, "Should only have gold label type (pipeline scripts only in gold data)"


class TestIntegration:
    """Integration tests using real data."""
    
    def test_real_data_pipeline(self):
        """Test the full pipeline with real data."""
        # Get inventory
        inventory = list_inventory()
        assert len(inventory) > 0, "Should have some inventory data"
        
        # Get first dataset ID
        test_id = inventory['id'].iloc[0]
        
        # Get pipeline script (only available in gold data)
        script = get_pipeline_script(id=test_id)
        assert len(script) > 0, "Should have a pipeline script"
        
        # Get lineages
        gold_lineages = get_lineages(id=test_id, label_type='gold')
        assert len(gold_lineages) > 0, "Should have lineage data"
        
        # Check if we have corresponding generated data
        try:
            generated_lineages = get_lineages(id=test_id, label_type='generated')
            # If we get here, test the batch evaluation
            gold_ordered, generated_ordered = prepare_batch_evaluation(gold_lineages, generated_lineages)
            assert len(gold_ordered) == len(generated_ordered), "Should have matching lengths"
        except ValueError:
            # It's ok if there's no corresponding generated data
            pass
    
    def test_data_consistency(self):
        """Test that inventory IDs match data file IDs."""
        inventory = list_inventory()
        inventory_ids = set(inventory['id'].tolist())
        
        # Check gold data
        data_path = Path(__file__).parent.parent / "slice" / "data" / "datasets"
        with open(data_path / GOLD, 'r') as f:
            gold_data = json.load(f)
        gold_ids = set(gold_data.keys())
        
        # Check that all inventory IDs exist in gold data
        missing_in_gold = inventory_ids - gold_ids
        assert len(missing_in_gold) == 0, f"IDs in inventory but not in gold data: {missing_in_gold}"