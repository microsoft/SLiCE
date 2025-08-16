# SLiCE: Schema Lineage Composite Evaluation

[![PyPI version](https://badge.fury.io/py/slice-score.svg)](https://badge.fury.io/py/slice-score)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: ArXiv](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2508.07179)

SLiCE is a Python package for evaluating schema lineage extraction accuracy by comparing model predictions with gold standards. It provides comprehensive metrics for assessing the quality of schema lineage extraction in data pipeline analysis. 

## Features

- **Component-wise Evaluation**: Separate scoring for source schema, source tables, transformations, and aggregations
- **Multiple Similarity Metrics**: BLEU scores, fuzzy matching, F1 scores, and AST-based similarity
- **Flexible Weighting**: Customizable weights for different components and metrics
- **Multi-language Support**: Handles Python, SQL, and C# code in transformations
- **Sample Data Module**: Built-in access to curated datasets for testing and demonstration
- **Batch Processing**: Parallel evaluation of multiple lineage pairs
- **Command Line Interface**: Easy-to-use CLI for quick evaluations

## Installation

### From PyPI (recommended)

```bash
pip install slice-score
```

### From Source

```bash
git clone https://github.com/microsoft/SLiCE.git
cd SLiCE
pip install -e .
```

### Development Installation

For development with all testing and linting tools:

```bash
git clone https://github.com/microsoft/SLiCE.git
cd SLiCE

# Using pip
pip install -e ".[dev]"

# Using uv (recommended - faster)
uv sync --extra dev
```

## Quick Start

### Python API

```python
from slice import SchemaLineageEvaluator

# Initialize evaluator
evaluator = SchemaLineageEvaluator()

# Example lineage data
predicted = {
    "source_schema": "cuisine_type",
    "source_table": "restaurants.ss",
    "transformation": "R.cuisine_type AS CuisineType", 
    "aggregation": "COUNT() GROUP BY restaurant_id"
}

ground_truth = {
    "source_schema": "cuisine_type",
    "source_table": "restaurants.ss", 
    "transformation": "R.cuisine_type AS CuisineType",
    "aggregation": ""
}

# Evaluate
results = evaluator.evaluate(predicted, ground_truth)
print(f"Overall Score: {results['overall']:.4f}")
```

### Command Line Interface

```bash
# Basic evaluation
slice-eval predicted.json ground_truth.json

# With custom weights
slice-eval --weights source_table=0.5,transformation=0.3,aggregation=0.2 predicted.json ground_truth.json

# Include metadata evaluation
slice-eval --metadata predicted.json ground_truth.json

# Save results to file
slice-eval predicted.json ground_truth.json --output results.txt
```

## Data Format

SLiCE expects lineage data as dictionaries with the following structure:

```json
{
    "source_schema": "column_name",
    "source_table": "table_references",
    "transformation": "transformation_logic",
    "aggregation": "aggregation_operations",
    "metadata": "additional_metadata (optional)"
}
```

## Evaluation Metrics

### Component Scores

- **Source Schema**: Exact match of schema/column names
- **Source Table**: F1 score + fuzzy matching of table references  
- **Transformation**: BLEU + weighted BLEU + AST similarity
- **Aggregation**: BLEU + weighted BLEU + AST similarity
- **Metadata**: BLEU + weighted BLEU + AST similarity (optional)

### Overall Score

The final score combines component scores using configurable weights:

```
Overall = format_correctness × source_schema × (
    w₁ × source_table_score + 
    w₂ × transformation_score + 
    w₃ × aggregation_score +
    w₄ × metadata_score  # if applicable
)
```

Default weights: `source_table=0.4, transformation=0.4, aggregation=0.2`

## Configuration

### Custom Weights

```python
# Component weights
weights = {
    'source_table': 0.5,
    'transformation': 0.3, 
    'aggregation': 0.2
}

# Metric weights for transformations
transformation_weights = {
    'bleu': 0.6,
    'weighted_bleu': 0.3,
    'ast': 0.1
}

evaluator = SchemaLineageEvaluator(
    weights=weights,
    transformation_weights=transformation_weights
)
```

### Language Support

```python
# Custom syntax and operators
evaluator = SchemaLineageEvaluator(
    sql_syntax={'SELECT', 'FROM', 'WHERE'},
    python_syntax={'def', 'class', 'import'},
    csharp_syntax={'using', 'namespace', 'class'}
)
```

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Basic evaluation with default settings
- `custom_weights.py`: Using custom weights and configurations
- `batch_evaluation.py`: Processing multiple lineage pairs
- `sample_data_usage.py`: Using package sample data for evaluation. 

## Testing

Run the test suite:

```bash
# Using pip
pip install -e ".[dev]"
pytest

# Using uv (recommended)
uv sync --extra dev
uv run pytest

# Run with coverage
uv run pytest --cov=slice

# Run specific test file
uv run pytest tests/test_schema_lineage_evaluator.py -v

# Code quality checks
uv run black slice/ tests/     # Format code
uv run flake8 slice/           # Lint code  
uv run mypy slice/             # Type checking
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SLiCE in your research, please cite:

```bibtex
@software{slice2025,
  title={SLiCE: Schema Lineage Composite Evaluation},
  author={Jiaqi Yin and Yi-Wei Chen and Meng-Lung Lee and Xiya Liu},
  year={2025},
  url={https://github.com/microsoft/SLiCE}
}

@misc{yin2025schemalineageextractionscale,
      title={Schema Lineage Extraction at Scale: Multilingual Pipelines, Composite Evaluation, and Language-Model Benchmarks}, 
      author={Jiaqi Yin and Yi-Wei Chen and Meng-Lung Lee and Xiya Liu},
      year={2025},
      eprint={2508.07179},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.07179}, 
}
```

## Support

- **Documentation**: [Link to documentation]
- **Issues**: [GitHub Issues](https://github.com/microsoft/SLiCE/issues)
- **Discussions**: [GitHub Discussions](https://github.com/microsoft/SLiCE/discussions)
