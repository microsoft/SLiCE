"""
Tests for CLI module.
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from slice.cli import format_results, load_json_file, main


class TestCLIHelpers:
    """Test helper functions in CLI module."""

    def test_load_json_file_valid(self):
        """Test loading valid JSON file."""
        test_data = {
            "source_schema": "test",
            "source_table": "test",
            "transformation": "test",
            "aggregation": "test",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            result = load_json_file(temp_file)
            assert result == test_data
        finally:
            os.unlink(temp_file)

    def test_load_json_file_not_found(self):
        """Test loading non-existent JSON file."""
        with pytest.raises(SystemExit):
            load_json_file("non_existent_file.json")

    def test_load_json_file_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name

        try:
            with pytest.raises(SystemExit):
                load_json_file(temp_file)
        finally:
            os.unlink(temp_file)

    def test_format_results_basic(self):
        """Test formatting evaluation results."""
        results = {
            "overall": 0.8567,
            "format": 1.0,
            "source_schema": 1.0,
            "source_table": 0.75,
            "transformation": 0.85,
            "aggregation": 0.90,
        }

        formatted = format_results(results)

        assert "EVALUATION RESULTS" in formatted
        assert "Overall Score: 0.8567" in formatted
        assert "Format Correctness: 1.0000" in formatted
        assert "Source Schema: 1.0000" in formatted
        assert "Source Table: 0.7500" in formatted
        assert "Transformation: 0.8500" in formatted
        assert "Aggregation: 0.9000" in formatted

    def test_format_results_with_all_key(self):
        """Test formatting results with 'all' key instead of 'overall'."""
        results = {
            "all": 0.8567,
            "format": 1.0,
            "source_schema": 1.0,
            "source_table": 0.75,
        }

        formatted = format_results(results)
        assert "Overall Score: 0.8567" in formatted

    def test_format_results_with_metadata(self):
        """Test formatting results with metadata component."""
        results = {
            "overall": 0.8567,
            "format": 1.0,
            "source_schema": 1.0,
            "source_table": 0.75,
            "transformation": 0.85,
            "aggregation": 0.90,
            "metadata": 0.88,
        }

        formatted = format_results(results)
        assert "Metadata: 0.8800" in formatted


class TestCLIMain:
    """Test main CLI function."""

    def test_main_basic_usage(self):
        """Test basic CLI usage."""
        # Create temporary test files
        predicted_data = {
            "source_schema": "test_schema",
            "source_table": "test_table",
            "transformation": "test_transform",
            "aggregation": "test_agg",
        }

        ground_truth_data = {
            "source_schema": "test_schema",
            "source_table": "test_table",
            "transformation": "test_transform",
            "aggregation": "test_agg",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(predicted_data, f)
            predicted_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ground_truth_data, f)
            ground_truth_file = f.name

        try:
            # Mock sys.argv to simulate command line arguments
            test_args = ["slice-eval", predicted_file, ground_truth_file]

            with patch("sys.argv", test_args):
                with patch("builtins.print") as mock_print:
                    main()

                    # Check that results were printed
                    printed_output = "\n".join(
                        [str(call.args[0]) for call in mock_print.call_args_list]
                    )
                    assert "EVALUATION RESULTS" in printed_output
                    assert "Overall Score:" in printed_output

        finally:
            os.unlink(predicted_file)
            os.unlink(ground_truth_file)

    def test_main_with_custom_weights(self):
        """Test CLI with custom weights."""
        predicted_data = {
            "source_schema": "test_schema",
            "source_table": "test_table",
            "transformation": "test_transform",
            "aggregation": "test_agg",
        }

        ground_truth_data = {
            "source_schema": "test_schema",
            "source_table": "test_table",
            "transformation": "test_transform",
            "aggregation": "test_agg",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(predicted_data, f)
            predicted_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ground_truth_data, f)
            ground_truth_file = f.name

        try:
            test_args = [
                "slice-eval",
                "--weights",
                "source_table=0.5,transformation=0.3,aggregation=0.2",
                predicted_file,
                ground_truth_file,
            ]

            with patch("sys.argv", test_args):
                with patch("builtins.print") as mock_print:
                    main()

                    printed_output = "\n".join(
                        [str(call.args[0]) for call in mock_print.call_args_list]
                    )
                    assert "EVALUATION RESULTS" in printed_output

        finally:
            os.unlink(predicted_file)
            os.unlink(ground_truth_file)

    def test_main_with_output_file(self):
        """Test CLI with output file."""
        predicted_data = {
            "source_schema": "test_schema",
            "source_table": "test_table",
            "transformation": "test_transform",
            "aggregation": "test_agg",
        }

        ground_truth_data = {
            "source_schema": "test_schema",
            "source_table": "test_table",
            "transformation": "test_transform",
            "aggregation": "test_agg",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(predicted_data, f)
            predicted_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ground_truth_data, f)
            ground_truth_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_file = f.name

        try:
            test_args = [
                "slice-eval",
                "--output",
                output_file,
                predicted_file,
                ground_truth_file,
            ]

            with patch("sys.argv", test_args):
                main()

                # Check that output file was created and contains results
                with open(output_file, "r") as f:
                    content = f.read()
                    assert "EVALUATION RESULTS" in content
                    assert "Overall Score:" in content

        finally:
            os.unlink(predicted_file)
            os.unlink(ground_truth_file)
            os.unlink(output_file)

    def test_main_invalid_weights_format(self):
        """Test CLI with invalid weights format."""
        predicted_data = {
            "source_schema": "test_schema",
            "source_table": "test_table",
            "transformation": "test_transform",
            "aggregation": "test_agg",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(predicted_data, f)
            predicted_file = f.name

        try:
            test_args = [
                "slice-eval",
                "--weights",
                "invalid_format",
                predicted_file,
                predicted_file,
            ]

            with patch("sys.argv", test_args):
                with pytest.raises(SystemExit):
                    main()

        finally:
            os.unlink(predicted_file)


if __name__ == "__main__":
    pytest.main([__file__])
